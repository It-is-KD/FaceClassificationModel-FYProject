import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.utils import class_weight
from pathlib import Path
import gc
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Union
import concurrent.futures
from tensorflow.keras.callbacks import Callback
import logging
import cv2
from PIL import Image
from tqdm import tqdm
import shutil
import signal
import sys
from model import ModelConfig, FaceRecognitionModel
from preprocessingUnit import FacePreprocessor

def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """Set up a logger with file and console handlers"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('TrainingPipeline')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"{log_dir}/training_{timestamp}.log")
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Timeout handler for long-running operations
class TimeoutError(Exception):
    pass

# Custom callback for timeout protection
class TimeoutCallback(Callback):
    def __init__(self, timeout_seconds: int = 3600):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.timeout_seconds:
            self.model.stop_training = True
            print(f"\nTraining stopped due to timeout after {elapsed_time:.2f} seconds")

# Custom callback to save checkpoint after each epoch
class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir, trainer):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.trainer = trainer
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        # Save model weights - use a single file that gets overwritten
        checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.keras")
        self.model.save(checkpoint_path)
        
        # Save training state
        state = {
            'epoch': epoch + 1,
            'optimizer_state': self.model.optimizer.get_weights() if hasattr(self.model.optimizer, 'get_weights') else None,
            'history': self.trainer.history.history if self.trainer.history else None,
            'current_lr': float(tf.keras.backend.get_value(self.model.optimizer.lr)) if hasattr(self.model.optimizer, 'lr') else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save state to JSON - Fix for JSON serialization error
        with open(os.path.join(self.checkpoint_dir, 'training_state.json'), 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            state_serializable = {}
            for k, v in state.items():
                if k == 'history' and v is not None:
                    state_serializable[k] = {key: [float(val) if isinstance(val, (np.number, np.integer)) else val for val in vals] 
                                            for key, vals in v.items()}
                elif k == 'optimizer_state' and v is not None:
                    # Skip optimizer state as it's not needed for resumption
                    continue
                else:
                    state_serializable[k] = v
            json.dump(state_serializable, f, indent=4)
        
        # Save latest checkpoint info
        latest_info = {'latest_epoch': epoch + 1, 'latest_checkpoint': checkpoint_path}
        with open(os.path.join(self.checkpoint_dir, 'latest_checkpoint.json'), 'w') as f:
            json.dump(latest_info, f, indent=4)

# Custom callback to monitor and save models when overfitting is detected
class OverfittingMonitorCallback(Callback):
    def __init__(self, model_dir: str, patience: int = 5, min_delta: float = 0.01):
        super().__init__()
        self.model_dir = model_dir
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.overfit_count = 0
        self.last_save_epoch = 0
        os.makedirs(model_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        train_loss = logs.get('loss')
        
        if val_loss is None or train_loss is None:
            return
        
        # Check if validation loss is improving
        val_loss_improved = val_loss < self.best_val_loss - self.min_delta
        if val_loss_improved:
            self.best_val_loss = val_loss
            self.overfit_count = 0
            
            # Save best validation model - overwrite existing file
            if epoch > 0:  # Skip first epoch
                model_path = os.path.join(self.model_dir, "best_val_model.keras")
                self.model.save(model_path)
                print(f"\nNew best validation model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
        
        # Check for overfitting (train loss decreasing but val loss increasing)
        train_loss_improved = train_loss < self.best_train_loss - self.min_delta
        
        if train_loss_improved and not val_loss_improved:
            self.overfit_count += 1
            
            # If we've seen signs of overfitting for 'patience' epochs, save the model
            if self.overfit_count >= self.patience and (epoch - self.last_save_epoch) >= self.patience:
                model_path = os.path.join(self.model_dir, "pre_overfit_model.keras")
                self.model.save(model_path)
                print(f"\nPotential overfitting detected at epoch {epoch+1}. Model saved as {model_path}")
                self.last_save_epoch = epoch
        
        # Update best train loss
        if train_loss_improved:
            self.best_train_loss = train_loss

# Custom dynamic learning rate scheduler
class DynamicLearningRateScheduler(Callback):
    def __init__(self, initial_lr=0.001, min_lr=1e-6, decay_factor=0.75, 
                 patience=3, cooldown=2, warmup_epochs=5, verbose=1):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.cooldown = cooldown
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        self.wait = 0
        self.best_loss = float('inf')
        self.cooldown_counter = 0
        self.in_cooldown = False
        
    def on_train_begin(self, logs=None):
        # Set initial learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)
        
    def on_epoch_begin(self, epoch, logs=None):
        # Implement warmup - gradually increase LR during initial epochs
        if epoch < self.warmup_epochs:
            # Linear warmup from 10% to 100% of initial_lr
            warmup_lr = self.initial_lr * (0.1 + 0.9 * (epoch / self.warmup_epochs))
            tf.keras.backend.set_value(self.model.optimizer.lr, warmup_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: Warmup LR set to {warmup_lr:.6f}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get('val_loss', logs.get('loss', 0))
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        # Skip LR adjustment during warmup
        if epoch < self.warmup_epochs:
            return
        
        # Check if in cooldown period
        if self.in_cooldown:
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.in_cooldown = False
            return
        
        # If loss improved, reset wait counter
        if current_loss < self.best_loss - 0.001:  # Small threshold to avoid minor fluctuations
            self.best_loss = current_loss
            self.wait = 0
        else:
            # If loss didn't improve, increment wait counter
            self.wait += 1
            
            # If waited for 'patience' epochs, reduce learning rate
            if self.wait >= self.patience:
                # Calculate new learning rate
                new_lr = max(current_lr * self.decay_factor, self.min_lr)
                
                # Only update if there's a significant change
                if new_lr < current_lr - 1e-8:
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch+1}: Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
                    
                    # Reset wait counter and enter cooldown
                    self.wait = 0
                    self.in_cooldown = True
                    self.cooldown_counter = self.cooldown

# Custom accuracy metric for face recognition
class FaceRecognitionAccuracy(Callback):
    def __init__(self, validation_data=None, batch_size=32):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size
        
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            return
            
        logs = logs or {}
        
        # Get validation data
        val_images, val_labels = self.validation_data
        
        # Get embeddings for validation images
        embeddings = []
        for i in range(0, len(val_images), self.batch_size):
            batch = val_images[i:min(i + self.batch_size, len(val_images))]
            batch_embeddings = self.model.predict(batch, verbose=0)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # For each sample, find the most similar other sample
        accuracy_count = 0
        total_comparisons = 0
        
        for i in range(len(val_labels)):
            # Get similarities for this sample (excluding self-comparison)
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            
            # Find most similar sample
            most_similar_idx = np.argmax(similarities)
            
            # Check if labels match
            if val_labels[i] == val_labels[most_similar_idx]:
                accuracy_count += 1
            
            total_comparisons += 1
        
        # Calculate accuracy
        val_accuracy = accuracy_count / total_comparisons if total_comparisons > 0 else 0
        
        # Add to logs
        logs['val_face_accuracy'] = val_accuracy
        
        print(f" - val_face_accuracy: {val_accuracy:.4f}")

def setup_tpu():
    """Set up TPU for TensorFlow"""
    try:
        # Detect TPUs
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f'Running on TPU: {tpu.cluster_spec().as_dict()}')
        
        # Connect to TPU system
        tf.config.experimental_connect_to_cluster(tpu)
        
        # Initialize TPU system
        tf.tpu.experimental.initialize_tpu_system(tpu)
        
        # Create distribution strategy
        strategy = tf.distribute.TPUStrategy(tpu)
        
        print(f"TPU Strategy created with {strategy.num_replicas_in_sync} replicas")
        return strategy
    
    except ValueError:
        print("No TPU detected, falling back to GPU/CPU")
        return None

# Custom cosine similarity metric for model evaluation
def cosine_similarity_metric(y_true, y_pred):
    # Normalize embeddings
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=1)
    
    # Calculate cosine similarity
    similarity = tf.matmul(y_pred_norm, y_pred_norm, transpose_b=True)
    
    # Create a mask for positive pairs (same class)
    mask_pos = tf.cast(tf.equal(tf.expand_dims(y_true, 1), tf.expand_dims(y_true, 0)), tf.float32)
    
    # Create a mask for negative pairs (different class)
    mask_neg = tf.cast(tf.not_equal(tf.expand_dims(y_true, 1), tf.expand_dims(y_true, 0)), tf.float32)
    
    # Calculate average similarity for positive and negative pairs
    pos_sim = tf.reduce_sum(similarity * mask_pos) / (tf.reduce_sum(mask_pos) + 1e-8)
    neg_sim = tf.reduce_sum(similarity * mask_neg) / (tf.reduce_sum(mask_neg) + 1e-8)
    
    # Return difference between positive and negative similarities
    return pos_sim - neg_sim

class FaceRecognitionTrainer:
    def __init__(self, 
                 train_dir: str,
                 val_dir: str,
                 test_dir: str,
                 output_dir: str,
                 model_config: Optional[ModelConfig] = None,
                 batch_size: int = 32,
                 max_samples_per_class: Optional[int] = None,
                 timeout_seconds: int = 14400,  # 4 hours default timeout
                 use_gpu: bool = True,
                 use_custom_cnn: bool = True,
                 use_tpu: bool = False,
                 embedding_dim: int = 512,
                 learning_rate: float = 0.001,
                 use_arcface: bool = True,
                 use_attention: bool = True):
        """
        Initialize the face recognition training pipeline with enhanced parameters
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            test_dir: Directory containing test data
            output_dir: Directory to save model and results
            model_config: Configuration for the model (optional)
            batch_size: Batch size for training
            max_samples_per_class: Maximum samples per class for balancing
            timeout_seconds: Maximum training time in seconds
            use_gpu: Whether to use GPU acceleration
            use_custom_cnn: Whether to use custom CNN architecture
            use_tpu: Whether to use TPU if available
            embedding_dim: Dimension of face embeddings
            learning_rate: Initial learning rate
            use_arcface: Whether to use ArcFace loss
            use_attention: Whether to use attention mechanism
        """
        # Set up directories
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_samples_per_class = max_samples_per_class
        self.timeout_seconds = timeout_seconds
        self.use_custom_cnn = use_custom_cnn
        self.embedding_dim = embedding_dim
        self.initial_learning_rate = learning_rate
        self.use_arcface = use_arcface
        self.use_attention = use_attention
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.model_dir = os.path.join(output_dir, 'models')
        self.log_dir = os.path.join(output_dir, 'logs')
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.report_dir = os.path.join(output_dir, 'reports')
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        
        for directory in [self.model_dir, self.log_dir, self.viz_dir, self.report_dir, self.checkpoint_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logger
        self.logger = setup_logger(self.log_dir)
        
        # Set up TPU if requested
        self.tpu_strategy = None
        if use_tpu:
            self.tpu_strategy = setup_tpu()
            if self.tpu_strategy:
                self.logger.info(f"TPU strategy initialized with {self.tpu_strategy.num_replicas_in_sync} replicas")
                # Adjust batch size for TPU (must be divisible by number of replicas)
                num_replicas = self.tpu_strategy.num_replicas_in_sync
                self.batch_size = (batch_size // num_replicas) * num_replicas
                self.logger.info(f"Adjusted batch size for TPU: {self.batch_size}")
            else:
                self.logger.warning("TPU requested but not available. Using GPU/CPU instead.")
        
        # Configure GPU if TPU is not available
        if not self.tpu_strategy:
            self.device = self._setup_gpu(use_gpu)
        else:
            self.device = 'tpu'
        
        # Initialize preprocessor with enhanced parameters
        self.preprocessor = FacePreprocessor(
            target_size=(224, 224),
            use_mtcnn=True,
            normalize_range=(-1, 1),
            device=self.device if self.device != 'tpu' else 'cpu',  # Use CPU for preprocessing if TPU is used
            confidence_threshold=0.85,
            margin_percent=0.25
        )
        
        # Initialize model with default or provided config
        if model_config is None:
            self.model_config = ModelConfig(
                input_shape=(224, 224, 3),
                embedding_dim=embedding_dim,
                use_pretrained=not use_custom_cnn,  # Use pretrained only if not using custom CNN
                base_model_type="custom" if use_custom_cnn else "resnet50",
                use_attention=use_attention,
                use_arcface=use_arcface,
                dropout_rate=0.5,
                l2_regularization=0.01,
                learning_rate=learning_rate
            )
        else:
            self.model_config = model_config
        
        # Initialize model (will be created in TPU scope if TPU is available)
        if self.tpu_strategy:
            with self.tpu_strategy.scope():
                self.model = FaceRecognitionModel(self.model_config)
        else:
            self.model = FaceRecognitionModel(self.model_config)
        
        # Dataset statistics
        self.class_counts = {}
        self.class_weights = None
        self.num_classes = 0
        
        # Training metrics
        self.history = None
        self.test_metrics = None
        
        # Resumption state
        self.start_epoch = 0
        self.is_resuming = False
        
        self.logger.info(f"Trainer initialized with:")
        self.logger.info(f"  - Train dir: {train_dir}")
        self.logger.info(f"  - Val dir: {val_dir}")
        self.logger.info(f"  - Test dir: {test_dir}")
        self.logger.info(f"  - Output dir: {output_dir}")
        self.logger.info(f"  - Batch size: {batch_size}")
        self.logger.info(f"  - Using custom CNN: {use_custom_cnn}")
        self.logger.info(f"  - Using ArcFace: {use_arcface}")
        self.logger.info(f"  - Using Attention: {use_attention}")
        self.logger.info(f"  - Embedding dimension: {embedding_dim}")
        self.logger.info(f"  - Initial learning rate: {learning_rate}")
        self.logger.info(f"  - Device: {self.device}")
        
        # Register signal handler for graceful interruption
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, sig, frame):
        """Handle keyboard interrupt (Ctrl+C)"""
        if self.interrupted:
            # If already interrupted once, exit immediately
            self.logger.warning("Forced exit. Training state may be inconsistent.")
            sys.exit(1)
        
        self.interrupted = True
        self.logger.warning("\nTraining interrupted! Saving checkpoint before exiting...")
        
        # Save checkpoint if model exists
        if hasattr(self, 'model') and self.model.model is not None:
            self._save_checkpoint("interrupt")
            # Also save a special interrupted model
            interrupted_model_path = os.path.join(self.model_dir, "interrupted_model.keras")
            self.model.save_model(interrupted_model_path)
            self.logger.info(f"Interrupted model saved to {interrupted_model_path}")
            self.logger.info("Checkpoint saved. You can resume training by running the script again.")
        
        sys.exit(0)
    
    def _save_checkpoint(self, checkpoint_type="regular"):
        """Save training checkpoint with proper JSON serialization"""
        # Save model weights
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_type}.keras")
        self.model.save_model(checkpoint_path)
        
        # Determine current epoch
        current_epoch = self.start_epoch
        if self.history and 'loss' in self.history.history:
            current_epoch += len(self.history.history['loss'])
        
        # Save training state
        state = {
            'epoch': current_epoch,
            'model_config': self.model_config.__dict__,
            'history': self.history.history if self.history else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save state to JSON with proper serialization
        with open(os.path.join(self.checkpoint_dir, 'training_state.json'), 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            state_serializable = {}
            for k, v in state.items():
                if k == 'history' and v is not None:
                    state_serializable[k] = {key: [float(val) if isinstance(val, (np.number, np.integer)) else val for val in vals] 
                                            for key, vals in v.items()}
                elif k == 'model_config':
                    state_serializable[k] = {key: float(val) if isinstance(val, (np.number, np.integer)) else val 
                                           for key, val in v.items()}
                else:
                    state_serializable[k] = v
            json.dump(state_serializable, f, indent=4)
        
        # Save latest checkpoint info
        latest_info = {'latest_epoch': int(current_epoch), 'latest_checkpoint': checkpoint_path}
        with open(os.path.join(self.checkpoint_dir, 'latest_checkpoint.json'), 'w') as f:
            json.dump(latest_info, f, indent=4)
    
    def _check_for_checkpoint(self):
        """Check if a checkpoint exists to resume training"""
        latest_checkpoint_file = os.path.join(self.checkpoint_dir, 'latest_checkpoint.json')
        
        if not os.path.exists(latest_checkpoint_file):
            return False
        
        try:
            with open(latest_checkpoint_file, 'r') as f:
                checkpoint_info = json.load(f)
            
            latest_epoch = checkpoint_info.get('latest_epoch', 0)
            latest_checkpoint = checkpoint_info.get('latest_checkpoint', '')
            
            if latest_epoch > 0 and os.path.exists(latest_checkpoint):
                self.logger.info(f"Found checkpoint at epoch {latest_epoch}")
                
                # Load training state
                state_file = os.path.join(self.checkpoint_dir, 'training_state.json')
                if os.path.exists(state_file):
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    
                    # Set start epoch
                    self.start_epoch = state.get('epoch', 0)
                    
                    # Load history if available
                    if 'history' in state and state['history']:
                        # Create a mock history object
                        class MockHistory:
                            pass
                        
                        self.history = MockHistory()
                        self.history.history = state['history']
                
                # Load model weights with proper error handling
                try:
                    # First attempt: Try loading with custom_objects for Lambda layers
                    custom_objects = {
                        'tf': tf,
                        'cosine_similarity_metric': cosine_similarity_metric
                    }
                    
                    # Try different loading approaches to handle various TensorFlow versions
                    try:
                        # Approach 1: Load with safe_mode=False (for newer TF versions)
                        self.model.load_model(latest_checkpoint, custom_objects=custom_objects, safe_mode=False)
                    except TypeError:
                        try:
                            # Approach 2: Load without safe_mode parameter (for older TF versions)
                            self.model.load_model(latest_checkpoint, custom_objects=custom_objects)
                        except Exception as e1:
                            self.logger.warning(f"Standard model loading failed: {str(e1)}")
                            try:
                                # Approach 3: Try loading the model directly with tf.keras.models.load_model
                                self.logger.warning("Attempting to load with tf.keras.models.load_model")
                                
                                # Load the model directly
                                if self.tpu_strategy:
                                    with self.tpu_strategy.scope():
                                        loaded_model = tf.keras.models.load_model(
                                            latest_checkpoint,
                                            custom_objects=custom_objects,
                                            compile=False  # Don't compile yet
                                        )
                                else:
                                    loaded_model = tf.keras.models.load_model(
                                        latest_checkpoint,
                                        custom_objects=custom_objects,
                                        compile=False  # Don't compile yet
                                    )
                                
                                # Replace the model in our FaceRecognitionModel instance
                                self.model.model = loaded_model
                            except Exception as e2:
                                self.logger.warning(f"Direct model loading failed: {str(e2)}")
                                # Approach 4: Create a new model with the same architecture and try to load weights
                                self.logger.warning("Recreating model architecture and attempting to load weights")
                                
                                # Recreate the model architecture
                                if self.tpu_strategy:
                                    with self.tpu_strategy.scope():
                                        self.model = FaceRecognitionModel(self.model_config)
                                        try:
                                            # Try loading weights directly
                                            self.model.model.load_weights(latest_checkpoint)
                                        except Exception as e3:
                                            self.logger.error(f"Weight loading failed: {str(e3)}")
                                            raise Exception("All loading approaches failed")
                                else:
                                    self.model = FaceRecognitionModel(self.model_config)
                                    try:
                                        # Try loading weights directly
                                        self.model.model.load_weights(latest_checkpoint)
                                    except Exception as e3:
                                        self.logger.error(f"Weight loading failed: {str(e3)}")
                                        raise Exception("All loading approaches failed")
                    
                    # Ensure model is compiled after loading
                    if self.tpu_strategy:
                        with self.tpu_strategy.scope():
                            # Set up arcface weights for the model if needed
                            if self.use_arcface:
                                self.model.arcface_weights = tf.Variable(
                                    tf.random.normal([self.num_classes, self.embedding_dim]),
                                    name='arcface_weights',
                                    trainable=True
                                )
                            self.model.compile_model()
                    else:
                        # Set up arcface weights for the model if needed
                        if self.use_arcface:
                            self.model.arcface_weights = tf.Variable(
                                tf.random.normal([self.num_classes, self.embedding_dim]),
                                name='arcface_weights',
                                trainable=True
                            )
                        self.model.compile_model()
                    
                    self.logger.info(f"Resumed model from {latest_checkpoint}")
                    
                    self.is_resuming = True
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error loading checkpoint: {str(e)}")
                    self.logger.warning("Creating a fresh model due to checkpoint loading failure")
                    
                    # Initialize a fresh model since loading failed
                    if self.tpu_strategy:
                        with self.tpu_strategy.scope():
                            self.model = FaceRecognitionModel(self.model_config)
                            # Set up arcface weights for the model if needed
                            if self.use_arcface:
                                self.model.arcface_weights = tf.Variable(
                                    tf.random.normal([self.num_classes, self.embedding_dim]),
                                    name='arcface_weights',
                                    trainable=True
                                )
                            self.model.compile_model()
                    else:
                        self.model = FaceRecognitionModel(self.model_config)
                        # Set up arcface weights for the model if needed
                        if self.use_arcface:
                            self.model.arcface_weights = tf.Variable(
                                tf.random.normal([self.num_classes, self.embedding_dim]),
                                name='arcface_weights',
                                trainable=True
                            )
                        self.model.compile_model()
                    
                    # Start from epoch 0
                    self.start_epoch = 0
                    self.is_resuming = False
                    return False
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
        
        return False

    def _setup_gpu(self, use_gpu: bool) -> str:
        """Configure GPU and return device string"""
        if not use_gpu:
            self.logger.info("GPU usage disabled. Using CPU.")
            return 'cpu'
        
        # Check TensorFlow GPU
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            self.logger.info(f"TensorFlow detected {len(tf_gpus)} GPU(s):")
            for i, gpu in enumerate(tf_gpus):
                self.logger.info(f"  {i+1}. {gpu.name}")
            
            # Configure memory growth to avoid OOM errors
            for gpu in tf_gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(f"Memory growth enabled for {gpu.name}")
                except RuntimeError as e:
                    self.logger.warning(f"Error setting memory growth: {e}")
        else:
            self.logger.warning("No TensorFlow GPU available.")
        
        # Check PyTorch GPU
        if torch.cuda.is_available():
            self.logger.info(f"PyTorch detected {torch.cuda.device_count()} GPU(s):")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"  {i+1}. {torch.cuda.get_device_name(i)}")
            return 'cuda:0'
        else:
            self.logger.warning("No PyTorch GPU available.")
            return 'cpu'
    
    def _process_image_with_timeout(self, image_path: str, timeout: int = 30) -> Optional[np.ndarray]:
        """Process a single image with timeout protection using threading instead of signals"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return None
            
            # Process image
            face = self.preprocessor.process_image(image)
            return face
        
        except Exception as e:
            self.logger.warning(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def _load_dataset_batch(self, 
                           data_dir: str, 
                           batch_paths: List[Tuple[str, str, int]]) -> Tuple[List[np.ndarray], List[int]]:
        """Load a batch of images with parallel processing and timeout protection"""
        images = []
        labels = []
        
        # Use a timeout per worker instead of signal-based timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_path = {}
            
            # Submit all tasks
            for subdir, img_file, label in batch_paths:
                img_path = os.path.join(data_dir, subdir, img_file)
                future = executor.submit(self._process_image_with_timeout, img_path)
                future_to_path[future] = label
            
            # Process results with timeout
            for future in concurrent.futures.as_completed(future_to_path):
                label = future_to_path[future]
                try:
                    # Add a timeout here instead of in the processing function
                    face = future.result(timeout=30)  # 30 seconds timeout
                    if face is not None and face.shape == (224, 224, 3):
                        images.append(face)
                        labels.append(label)
                except concurrent.futures.TimeoutError:
                    self.logger.warning(f"Processing timed out for an image")
                except Exception as e:
                    self.logger.warning(f"Error in batch loading: {str(e)}")
        
        return images, labels
    
    def load_dataset(self, data_dir: str, is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset with memory-efficient batch processing and class balancing
        
                Args:
            data_dir: Directory containing the dataset
            is_training: Whether this is the training dataset (for class balancing)
            
        Returns:
            Tuple of (images, labels)
        """
        self.logger.info(f"Loading dataset from: {data_dir}")
        
        if not os.path.exists(data_dir):
            self.logger.error(f"Directory not found: {data_dir}")
            return np.array([]), np.array([])
        
        # Get all image paths and labels
        all_image_paths = []
        person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Create label mapping
        label_to_id = {person_dir: idx for idx, person_dir in enumerate(sorted(person_dirs))}
        self.num_classes = len(label_to_id)
        
        # Count samples per class
        class_samples = {label: 0 for label in range(self.num_classes)}
        
        for person_dir in person_dirs:
            person_path = os.path.join(data_dir, person_dir)
            label = label_to_id[person_dir]
            
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Count samples
            class_samples[label] = len(image_files)
            
            # Apply class balancing for training data
            if is_training and self.max_samples_per_class is not None:
                if len(image_files) > self.max_samples_per_class:
                    # Randomly select max_samples_per_class images
                    np.random.shuffle(image_files)
                    image_files = image_files[:self.max_samples_per_class]
            
            # Add image paths with labels
            for img_file in image_files:
                all_image_paths.append((person_dir, img_file, label))
        
        # Save class distribution for training data
        if is_training:
            self.class_counts = class_samples
            
            # Calculate class weights for imbalanced dataset
            labels_list = [label for _, _, label in all_image_paths]
            self.class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels_list),
                y=labels_list
            )
            
            # Log class distribution
            self.logger.info(f"Class distribution: {self.class_counts}")
            
            # Visualize class distribution
            self._visualize_class_distribution(class_samples)
        
        # Process images in batches to manage memory
        self.logger.info(f"Processing {len(all_image_paths)} images in batches")
        
        all_images = []
        all_labels = []
        
        # Process in batches of 100 images
        batch_size = 100
        for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Loading batches"):
            batch_paths = all_image_paths[i:i+batch_size]
            
            # Load batch with parallel processing
            images, labels = self._load_dataset_batch(data_dir, batch_paths)
            
            all_images.extend(images)
            all_labels.extend(labels)
            
            # Force garbage collection to free memory
            gc.collect()
        
        if not all_images:
            self.logger.warning(f"No valid images found in {data_dir}")
            return np.array([]), np.array([])
        
        self.logger.info(f"Loaded {len(all_images)} images with {len(set(all_labels))} unique identities")
        
        # Convert to numpy arrays
        return np.stack(all_images), np.array(all_labels)
    
    def _visualize_class_distribution(self, class_samples: Dict[int, int]) -> None:
        """Visualize and save class distribution"""
        plt.figure(figsize=(12, 6))
        
        # Sort by class index
        sorted_items = sorted(class_samples.items())
        classes = [str(cls) for cls, _ in sorted_items]
        counts = [count for _, count in sorted_items]
        
        # Plot class distribution
        plt.bar(classes, counts)
        plt.title('Class Distribution in Training Data')
        plt.xlabel('Class ID')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.viz_dir, 'class_distribution.png')
        plt.savefig(viz_path)
        plt.close()
        
        self.logger.info(f"Class distribution visualization saved to {viz_path}")
    
    def _prepare_one_hot_labels(self, labels: np.ndarray) -> np.ndarray:
        """Convert integer labels to one-hot encoded format for ArcFace loss"""
        # Find the actual number of classes based on the maximum label value
        max_label = np.max(labels)
        actual_num_classes = max_label + 1  # +1 because labels are zero-indexed
        
        # Make sure num_classes is at least as large as the maximum label
        if self.num_classes < actual_num_classes:
            self.logger.warning(f"Updating num_classes from {self.num_classes} to {actual_num_classes} based on label data")
            self.num_classes = actual_num_classes
        
        # Use TensorFlow's to_categorical function
        return tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
    
    def train_model(self, epochs: int = 100) -> None:
        """Train the face recognition model with enhanced monitoring and visualization"""
        self.logger.info("Starting model training pipeline...")
        
        # Check for existing checkpoint to resume training
        resumed = self._check_for_checkpoint()
        if resumed:
            self.logger.info(f"Resuming training from epoch {self.start_epoch}")
        else:
            self.logger.info("Starting new training session")
        
        # Load datasets with class balancing for training
        self.logger.info("Loading training dataset...")
        train_images, train_labels = self.load_dataset(self.train_dir, is_training=True)
        
        self.logger.info("Loading validation dataset...")
        val_images, val_labels = self.load_dataset(self.val_dir)
        
        if len(train_images) == 0 or len(val_images) == 0:
            self.logger.error("Insufficient data for training. Exiting.")
            return
        
        # Prepare one-hot encoded labels for ArcFace if needed
        if self.use_arcface:
            train_one_hot = self._prepare_one_hot_labels(train_labels)
            val_one_hot = self._prepare_one_hot_labels(val_labels)
        
        # Define callbacks with timeout protection, checkpoint saving, and overfitting monitoring
        callbacks = [
            # Model checkpoint to save best model - overwrite the same file
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_loss'
            ),
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                monitor='val_loss',
                restore_best_weights=True,
                verbose=1
            ),
            # Dynamic learning rate scheduler
            DynamicLearningRateScheduler(
                initial_lr=self.initial_learning_rate,
                min_lr=1e-7,
                decay_factor=0.7,
                patience=5,
                cooldown=2,
                warmup_epochs=3,
                verbose=1
            ),
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            # CSV Logger for detailed metrics
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.log_dir, 'training_log.csv'),
                append=resumed  # Append if resuming training
            ),
            # Custom timeout callback
            TimeoutCallback(timeout_seconds=self.timeout_seconds),
            # Custom checkpoint callback for regular saving - now overwriting the same file
            CheckpointCallback(
                checkpoint_dir=self.checkpoint_dir,
                trainer=self
            ),
            # Custom overfitting monitor callback - modify to overwrite files
            OverfittingMonitorCallback(
                model_dir=self.model_dir,
                patience=3,  # Detect overfitting after 3 epochs
                min_delta=0.005  # Minimum change to be considered significant
            ),
            # Custom face recognition accuracy metric
            FaceRecognitionAccuracy(
                validation_data=(val_images, val_labels),
                batch_size=self.batch_size
            )
        ]

        # Compile model with appropriate loss function if not resuming
        if not resumed:
            # If using TPU, we need to compile the model within the strategy scope
            if self.tpu_strategy:
                with self.tpu_strategy.scope():
                    # Set up arcface weights for the model
                    if self.use_arcface:
                        self.model.arcface_weights = tf.Variable(
                            tf.random.normal([self.num_classes, self.embedding_dim]),
                            name='arcface_weights',
                            trainable=True
                        )
                    self.model.compile_model()
            else:
                # Set up arcface weights for the model
                if self.use_arcface:
                    self.model.arcface_weights = tf.Variable(
                        tf.random.normal([self.num_classes, self.embedding_dim]),
                        name='arcface_weights',
                        trainable=True
                    )
                self.model.compile_model()
        
        # Log model summary
        self.logger.info("Model architecture:")
        self.model.model.summary(print_fn=self.logger.info)
        
        # Save model architecture visualization with error handling
        try:
            tf.keras.utils.plot_model(
                self.model.model,
                to_file=os.path.join(self.viz_dir, 'model_architecture.png'),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True
            )
            self.logger.info("Model architecture visualization saved")
        except Exception as e:
            self.logger.warning(f"Could not generate model architecture visualization: {str(e)}")
            self.logger.warning("This is a non-critical error and training will continue.")
        
        # Train the model with TPU/GPU acceleration and class weights
        remaining_epochs = epochs - self.start_epoch
        self.logger.info(f"Starting training for {remaining_epochs} more epochs (total: {epochs}) with batch size {self.batch_size}...")
        start_time = time.time()
        
        # Convert class weights to dictionary format for Keras
        class_weight_dict = {i: float(weight) for i, weight in enumerate(self.class_weights)} if self.class_weights is not None else None
        
        try:
            # Prepare data for TPU if needed
            if self.tpu_strategy:
                # Convert data to TensorFlow datasets for better TPU performance
                if self.use_arcface:
                    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_one_hot))
                    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_one_hot))
                else:
                    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
                
                train_dataset = train_dataset.shuffle(buffer_size=10000).batch(self.batch_size, drop_remainder=True)
                train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
                
                val_dataset = val_dataset.batch(self.batch_size, drop_remainder=True)
                val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
                
                # Train with TPU
                with self.tpu_strategy.scope():
                    history = self.model.model.fit(
                        train_dataset,
                        epochs=epochs,
                        initial_epoch=self.start_epoch,
                        validation_data=val_dataset,
                        callbacks=callbacks
                    )
            else:
                # Use GPU if available
                with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                    # Prepare training data based on loss function
                    if self.use_arcface:
                        train_data = (train_images, train_one_hot)
                        validation_data = (val_images, val_one_hot)
                    else:
                        train_data = (train_images, train_labels)
                        validation_data = (val_images, val_labels)
                    
                    history = self.model.model.fit(
                        train_data[0],
                        train_data[1],
                        validation_data=validation_data,
                        epochs=epochs,
                        initial_epoch=self.start_epoch,
                        batch_size=self.batch_size,
                        callbacks=callbacks,
                        class_weight=class_weight_dict,
                        verbose=1
                    )
            
            # Merge history if resuming
            if resumed and self.history:
                # Combine previous history with new history
                combined_history = {}
                for key in history.history:
                    if key in self.history.history:
                        combined_history[key] = self.history.history[key] + history.history[key]
                    else:
                        combined_history[key] = history.history[key]
                
                # Create a new history object with combined data
                class CombinedHistory:
                    pass
                
                combined_hist_obj = CombinedHistory()
                combined_hist_obj.history = combined_history
                self.history = combined_hist_obj
            else:
                self.history = history
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, 'final_model.keras')
            self.model.save_model(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")
            
            # Plot and save training history
            self._plot_training_history()
            
            # Evaluate model on test set
            self.evaluate_model()
            
            # Generate comprehensive training report
            self._generate_training_report(training_time)
            
            # Clean up checkpoints if training completed successfully
            self._cleanup_checkpoints()
            
        except KeyboardInterrupt:
            self.logger.warning("\nTraining interrupted by user!")
            # Save checkpoint
            self._save_checkpoint("interrupt")
            # Also save a special interrupted model
            interrupted_model_path = os.path.join(self.model_dir, "interrupted_model.keras")
            self.model.save_model(interrupted_model_path)
            self.logger.info(f"Interrupted model saved to {interrupted_model_path}")
            self.logger.info("Checkpoint saved. You can resume training by running the script again.")
            return
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            # Save emergency checkpoint
            emergency_model_path = os.path.join(self.model_dir, "emergency_model.keras")
            try:
                self.model.save_model(emergency_model_path)
                self.logger.info(f"Emergency model saved to {emergency_model_path}")
            except Exception as save_error:
                self.logger.error(f"Failed to save emergency model: {str(save_error)}")
            raise
    
    def _cleanup_checkpoints(self):
        """Clean up intermediate checkpoints after successful training"""
        try:
            # Keep only the latest checkpoint and best model
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                              if f.startswith('checkpoint_epoch_') and f.endswith('.keras')]
            
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Keep the latest checkpoint
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                for checkpoint in checkpoint_files:
                    if checkpoint != latest_checkpoint:
                        os.remove(os.path.join(self.checkpoint_dir, checkpoint))
                        
            self.logger.info("Cleaned up intermediate checkpoints")
        except Exception as e:
            self.logger.warning(f"Error cleaning up checkpoints: {str(e)}")
    
    def _plot_training_history(self) -> None:
        """Plot and save detailed training history with accuracy metrics"""
        if self.history is None:
            self.logger.warning("No training history available to plot")
            return
        
        # Create figure with multiple subplots
        plt.figure(figsize=(15, 15))
        
        # Plot training & validation loss
        plt.subplot(3, 1, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot accuracy metrics if available
        if 'accuracy' in self.history.history or 'val_accuracy' in self.history.history:
            plt.subplot(3, 1, 2)
            if 'accuracy' in self.history.history:
                plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.history.history:
                plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot face recognition accuracy if available
        if 'val_face_accuracy' in self.history.history:
            plt.subplot(3, 1, 3)
            plt.plot(self.history.history['val_face_accuracy'], label='Face Recognition Accuracy')
            plt.title('Face Recognition Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
        # Otherwise plot learning rate
        elif 'lr' in self.history.history:
            plt.subplot(3, 1, 3)
            plt.semilogy(self.history.history['lr'], label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate (log scale)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        history_path = os.path.join(self.viz_dir, 'training_history.png')
        plt.savefig(history_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Training history plot saved to {history_path}")
        
        # Save history data as CSV for further analysis
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(os.path.join(self.report_dir, 'training_history.csv'), index=False)
        
        # Plot overfitting analysis
        self._plot_overfitting_analysis()
    
    def _plot_overfitting_analysis(self) -> None:
        """Plot and save overfitting analysis"""
        if self.history is None or 'loss' not in self.history.history or 'val_loss' not in self.history.history:
            return
            
        # Calculate the gap between training and validation loss
        train_loss = np.array(self.history.history['loss'])
        val_loss = np.array(self.history.history['val_loss'])
        gap = val_loss - train_loss
        
        # Plot the gap
        plt.figure(figsize=(12, 6))
        plt.plot(gap, label='Validation - Training Loss')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title('Overfitting Analysis: Gap Between Validation and Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Gap')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add annotations for potential overfitting regions
        threshold = 0.05  # Threshold for considering overfitting
        for i in range(1, len(gap)):
            if gap[i] > threshold and gap[i] > gap[i-1]:
                plt.annotate('Potential Overfitting', 
                            xy=(i, gap[i]),
                            xytext=(i, gap[i] + 0.05),
                            arrowprops=dict(facecolor='red', shrink=0.05),
                            horizontalalignment='center')
        
        plt.tight_layout()
        overfitting_path = os.path.join(self.viz_dir, 'overfitting_analysis.png')
        plt.savefig(overfitting_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Overfitting analysis saved to {overfitting_path}")
    
    def evaluate_model(self) -> Dict:
        """
        Evaluate model on test set with comprehensive metrics
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Evaluating model on test set...")
        
        # Load test dataset
        test_images, test_labels = self.load_dataset(self.test_dir)
        
        if len(test_images) == 0:
            self.logger.error("No test data available for evaluation")
            return {}
        
        # Get embeddings for test images
        if self.tpu_strategy:
            # For TPU, process in batches
            test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
            test_dataset = test_dataset.batch(self.batch_size)
            
            embeddings_list = []
            with self.tpu_strategy.scope():
                for batch in test_dataset:
                    batch_embeddings = self.model.model.predict(batch)
                    embeddings_list.append(batch_embeddings)
                
                embeddings = np.vstack(embeddings_list)
        else:
            # For GPU/CPU
            embeddings = self.model.get_embeddings(test_images)
        
        # Calculate metrics
        metrics = {}
        
        # Basic evaluation with model's evaluate method
        if self.tpu_strategy:
            # For TPU
            if self.use_arcface:
                test_one_hot = self._prepare_one_hot_labels(test_labels)
                test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_one_hot))
            else:
                test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                
            test_dataset = test_dataset.batch(self.batch_size, drop_remainder=True)
            
            with self.tpu_strategy.scope():
                test_loss = self.model.model.evaluate(test_dataset)
        else:
            # For GPU/CPU
            if self.use_arcface:
                test_one_hot = self._prepare_one_hot_labels(test_labels)
                test_loss = self.model.evaluate((test_images, test_one_hot))
            else:
                test_loss = self.model.evaluate((test_images, test_labels))
            
        metrics['test_loss'] = test_loss
        
        # Perform verification evaluation
        verification_metrics = self._evaluate_verification(embeddings, test_labels)
        metrics.update(verification_metrics)
        
        # Calculate face recognition accuracy
        recognition_accuracy = self._calculate_recognition_accuracy(embeddings, test_labels)
        metrics['recognition_accuracy'] = recognition_accuracy
        
        # Log metrics
        self.logger.info("Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value}")
        
        # Save metrics
        self.test_metrics = metrics
        
        # Generate evaluation visualizations
        self._visualize_embeddings(embeddings, test_labels)
        
        return metrics
    
    def _calculate_recognition_accuracy(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate face recognition accuracy using nearest neighbor classification
        
        Args:
            embeddings: Face embeddings
            labels: Ground truth labels
            
        Returns:
            Recognition accuracy
        """
        # Calculate cosine similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if i != j:  # Skip self-comparison
                    similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        
        # For each face, find the most similar face and check if labels match
        correct = 0
        total = 0
        
        for i in range(len(labels)):
            # Get similarities for this sample (excluding self-comparison)
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            
            # Find most similar sample
            most_similar_idx = np.argmax(similarities)
            
            # Check if labels match
            if labels[i] == labels[most_similar_idx]:
                correct += 1
            
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_verification(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate face verification performance with enhanced metrics
        
        Args:
            embeddings: Face embeddings
            labels: Ground truth labels
            
        Returns:
            Dictionary of verification metrics
        """
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        
        # Generate pairs for verification
        num_samples = len(embeddings)
        
        # Generate positive pairs (same identity)
        positive_pairs = []
        positive_labels = []
        
        # Generate negative pairs (different identity)
        negative_pairs = []
        negative_labels = []
        
        # Limit the number of pairs to avoid memory issues
        max_pairs = min(5000, num_samples * 10)
        
        # Generate positive pairs
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) >= 2:
                # Create pairs using integer indices
                pairs = []
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        pairs.append((indices[i], indices[j]))
                
                # Convert to numpy array if needed
                pairs = np.array(pairs)
                
                # Randomly select pairs if too many
                if len(pairs) > max_pairs // len(unique_labels):
                    np.random.shuffle(pairs)
                    pairs = pairs[:max_pairs // len(unique_labels)]
                
                for pair in pairs:
                    # Ensure indices are integers
                    idx1, idx2 = int(pair[0]), int(pair[1])
                    positive_pairs.append((embeddings[idx1], embeddings[idx2]))
                    positive_labels.append(1)
        
        # Generate negative pairs
        count = 0
        while count < len(positive_pairs):
            idx1 = np.random.randint(0, num_samples)
            idx2 = np.random.randint(0, num_samples)
            
            if labels[idx1] != labels[idx2]:
                negative_pairs.append((embeddings[idx1], embeddings[idx2]))
                negative_labels.append(0)
                count += 1
        
        # Combine pairs and compute distances
        all_pairs = positive_pairs + negative_pairs
        all_labels = np.array(positive_labels + negative_labels)
        
        # Compute cosine similarity
        similarities = []
        for emb1, emb2 in all_pairs:
            # Handle zero-norm vectors
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find best threshold
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        
        # Calculate accuracy at best threshold
        predictions = (similarities >= best_threshold).astype(int)
        accuracy = np.mean(predictions == all_labels)
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(all_labels, similarities)
        pr_auc = auc(recall, precision)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        roc_path = os.path.join(self.viz_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300)
        plt.close()
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        pr_path = os.path.join(self.viz_dir, 'precision_recall_curve.png')
        plt.savefig(pr_path, dpi=300)
        plt.close()
        
        # Plot similarity distribution
        plt.figure(figsize=(10, 8))
        plt.hist(similarities[all_labels == 1], bins=50, alpha=0.5, label='Same Identity')
        plt.hist(similarities[all_labels == 0], bins=50, alpha=0.5, label='Different Identity')
        plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Threshold: {best_threshold:.3f}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.title('Similarity Distribution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        sim_path = os.path.join(self.viz_dir, 'similarity_distribution.png')
        plt.savefig(sim_path, dpi=300)
        plt.close()
        
        return {
            'verification_auc': float(roc_auc),
            'verification_accuracy': float(accuracy),
            'best_threshold': float(best_threshold),
            'precision_recall_auc': float(pr_auc)
        }
    
    def _visualize_embeddings(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Visualize embeddings using t-SNE with improved visualization"""
        try:
            from sklearn.manifold import TSNE
            
            # Limit the number of samples for t-SNE to avoid memory issues
            max_samples = min(2000, len(embeddings))
            if len(embeddings) > max_samples:
                # Make sure indices are integers
                indices = np.random.choice(len(embeddings), max_samples, replace=False).astype(int)
                vis_embeddings = embeddings[indices]
                vis_labels = labels[indices]
            else:
                vis_embeddings = embeddings
                vis_labels = labels
            
            # Apply t-SNE
            self.logger.info("Computing t-SNE projection of embeddings...")
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            embeddings_2d = tsne.fit_transform(vis_embeddings)
            
            # Plot t-SNE visualization
            plt.figure(figsize=(12, 10))
            
            # Limit to 20 classes for clarity in visualization
            unique_labels = np.unique(vis_labels)
            if len(unique_labels) > 20:
                selected_labels = np.random.choice(unique_labels, 20, replace=False)
                # Create a boolean mask
                mask = np.zeros(len(vis_labels), dtype=bool)
                for label in selected_labels:
                    mask = mask | (vis_labels == label)
                
                embeddings_2d = embeddings_2d[mask]
                vis_labels = vis_labels[mask]
            
            # Create scatter plot with different colors for each class
            scatter = plt.scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1], 
                c=vis_labels, 
                cmap='tab20', 
                alpha=0.7,
                s=50
            )
            
            # Add legend
            legend1 = plt.legend(*scatter.legend_elements(),
                                title="Classes", loc="upper right")
            plt.gca().add_artist(legend1)
            
            plt.title('t-SNE Visualization of Face Embeddings')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            tsne_path = os.path.join(self.viz_dir, 'tsne_embeddings.png')
            plt.savefig(tsne_path, dpi=300)
            plt.close()
            
            self.logger.info(f"t-SNE visualization saved to {tsne_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating t-SNE visualization: {str(e)}")
    
    def _generate_training_report(self, training_time: float) -> None:
        """Generate comprehensive training report in HTML and JSON formats with proper serialization"""
        if self.history is None:
            self.logger.warning("No training history available for report generation")
            return
        
        # Create report data
        report_data = {
            "training_info": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "training_time_seconds": float(training_time),
                "training_time_minutes": float(training_time / 60),
                "epochs_completed": len(self.history.history['loss']),
                "batch_size": self.batch_size,
                "num_classes": self.num_classes,
                "class_balancing": self.max_samples_per_class is not None,
                "resumed_from_checkpoint": self.is_resuming,
                "start_epoch": self.start_epoch,
                "device": self.device
            },
            "dataset_info": {
                "train_dir": self.train_dir,
                "val_dir": self.val_dir,
                "test_dir": self.test_dir,
                "class_counts": {str(k): int(v) for k, v in self.class_counts.items()}  # Convert keys to strings
            },
            "model_info": {
                "architecture": "Custom CNN" if self.use_custom_cnn else "ResNet50",
                "embedding_dim": self.embedding_dim,
                "use_attention": self.use_attention,
                "use_arcface": self.use_arcface,
                "initial_learning_rate": float(self.initial_learning_rate)
            },
            "training_history": {
                "final_train_loss": float(self.history.history['loss'][-1]),
                "final_val_loss": float(self.history.history['val_loss'][-1]),
                "best_val_loss": float(min(self.history.history['val_loss'])),
                "best_epoch": int(np.argmin(self.history.history['val_loss']) + 1)
            }
        }
        
        # Add accuracy metrics if available
        if 'accuracy' in self.history.history:
            report_data["training_history"]["final_train_accuracy"] = float(self.history.history['accuracy'][-1])
        if 'val_accuracy' in self.history.history:
            report_data["training_history"]["final_val_accuracy"] = float(self.history.history['val_accuracy'][-1])
            report_data["training_history"]["best_val_accuracy"] = float(max(self.history.history['val_accuracy']))
        if 'val_face_accuracy' in self.history.history:
            report_data["training_history"]["final_face_accuracy"] = float(self.history.history['val_face_accuracy'][-1])
            report_data["training_history"]["best_face_accuracy"] = float(max(self.history.history['val_face_accuracy']))
        
        # Add evaluation metrics if available
        if self.test_metrics:
            # Ensure all values are JSON serializable
            report_data["evaluation_metrics"] = {}
            for k, v in self.test_metrics.items():
                if isinstance(v, (np.number, np.integer, np.floating)):
                    report_data["evaluation_metrics"][k] = float(v)
                else:
                    report_data["evaluation_metrics"][k] = v
        
        # Save as JSON
        json_path = os.path.join(self.report_dir, 'training_report.json')
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        # Generate HTML report
        html_path = os.path.join(self.report_dir, 'training_report.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Recognition Model Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; padding: 20px; border-radius: 5px; background-color: #f8f9fa; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .metric {{ font-weight: bold; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
                .image-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .image-item {{ flex: 1; min-width: 300px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Face Recognition Model Training Report</h1>
                
                <div class="section">
                    <h2>Training Information</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Date</td><td>{report_data['training_info']['date']}</td></tr>
                        <tr><td>Training Time</td><td>{report_data['training_info']['training_time_minutes']:.2f} minutes</td></tr>
                        <tr><td>Epochs Completed</td><td>{report_data['training_info']['epochs_completed']}</td></tr>
                        <tr><td>Resumed from Checkpoint</td><td>{'Yes' if report_data['training_info']['resumed_from_checkpoint'] else 'No'}</td></tr>
                        <tr><td>Start Epoch</td><td>{report_data['training_info']['start_epoch']}</td></tr>
                        <tr><td>Batch Size</td><td>{report_data['training_info']['batch_size']}</td></tr>
                        <tr><td>Number of Classes</td><td>{report_data['training_info']['num_classes']}</td></tr>
                        <tr><td>Class Balancing</td><td>{'Enabled' if report_data['training_info']['class_balancing'] else 'Disabled'}</td></tr>
                        <tr><td>Device</td><td>{report_data['training_info']['device']}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Model Information</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Architecture</td><td>{report_data['model_info']['architecture']}</td></tr>
                        <tr><td>Embedding Dimension</td><td>{report_data['model_info']['embedding_dim']}</td></tr>
                        <tr><td>Attention Mechanism</td><td>{'Enabled' if report_data['model_info']['use_attention'] else 'Disabled'}</td></tr>
                        <tr><td>ArcFace Loss</td><td>{'Enabled' if report_data['model_info']['use_arcface'] else 'Disabled'}</td></tr>
                        <tr><td>Initial Learning Rate</td><td>{report_data['model_info']['initial_learning_rate']}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Training Results</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Final Training Loss</td><td>{report_data['training_history']['final_train_loss']:.6f}</td></tr>
                        <tr><td>Final Validation Loss</td><td>{report_data['training_history']['final_val_loss']:.6f}</td></tr>
                        <tr><td>Best Validation Loss</td><td>{report_data['training_history']['best_val_loss']:.6f}</td></tr>
                        <tr><td>Best Epoch</td><td>{report_data['training_history']['best_epoch']}</td></tr>
        """
        
        # Add accuracy metrics if available
        if 'final_train_accuracy' in report_data['training_history']:
            html_content += f"<tr><td>Final Training Accuracy</td><td>{report_data['training_history']['final_train_accuracy']:.4f}</td></tr>\n"
        if 'final_val_accuracy' in report_data['training_history']:
            html_content += f"<tr><td>Final Validation Accuracy</td><td>{report_data['training_history']['final_val_accuracy']:.4f}</td></tr>\n"
        if 'best_val_accuracy' in report_data['training_history']:
            html_content += f"<tr><td>Best Validation Accuracy</td><td>{report_data['training_history']['best_val_accuracy']:.4f}</td></tr>\n"
        if 'final_face_accuracy' in report_data['training_history']:
            html_content += f"<tr><td>Final Face Recognition Accuracy</td><td>{report_data['training_history']['final_face_accuracy']:.4f}</td></tr>\n"
        if 'best_face_accuracy' in report_data['training_history']:
            html_content += f"<tr><td>Best Face Recognition Accuracy</td><td>{report_data['training_history']['best_face_accuracy']:.4f}</td></tr>\n"
        
        # Add evaluation metrics if available
        if 'evaluation_metrics' in report_data:
            html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Evaluation Metrics</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for metric_name, metric_value in report_data['evaluation_metrics'].items():
                if isinstance(metric_value, (int, float)):
                    html_content += f"<tr><td>{metric_name}</td><td>{metric_value:.6f}</td></tr>\n"
                else:
                    html_content += f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>\n"
        
        # Add visualizations
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    <div class="image-container">
        """
        
        # Add training history plot
        if os.path.exists(os.path.join(self.viz_dir, 'training_history.png')):
            html_content += """
                        <div class="image-item">
                            <h3>Training History</h3>
                            <img src="../visualizations/training_history.png" alt="Training History">
                        </div>
            """
        
        # Add overfitting analysis plot
        if os.path.exists(os.path.join(self.viz_dir, 'overfitting_analysis.png')):
            html_content += """
                        <div class="image-item">
                            <h3>Overfitting Analysis</h3>
                            <img src="../visualizations/overfitting_analysis.png" alt="Overfitting Analysis">
                        </div>
            """
        
        # Add class distribution plot
        if os.path.exists(os.path.join(self.viz_dir, 'class_distribution.png')):
            html_content += """
                        <div class="image-item">
                            <h3>Class Distribution</h3>
                            <img src="../visualizations/class_distribution.png" alt="Class Distribution">
                        </div>
            """
        
        # Add model architecture plot
        if os.path.exists(os.path.join(self.viz_dir, 'model_architecture.png')):
            html_content += """
                        <div class="image-item">
                            <h3>Model Architecture</h3>
                            <img src="../visualizations/model_architecture.png" alt="Model Architecture">
                        </div>
            """
        
        # Add t-SNE plot
        if os.path.exists(os.path.join(self.viz_dir, 'tsne_embeddings.png')):
            html_content += """
                        <div class="image-item">
                            <h3>t-SNE Embedding Visualization</h3>
                            <img src="../visualizations/tsne_embeddings.png" alt="t-SNE Embeddings">
                        </div>
            """
        
        # Add ROC curve
        if os.path.exists(os.path.join(self.viz_dir, 'roc_curve.png')):
            html_content += """
                        <div class="image-item">
                            <h3>ROC Curve</h3>
                            <img src="../visualizations/roc_curve.png" alt="ROC Curve">
                        </div>
            """
        
        # Add precision-recall curve
        if os.path.exists(os.path.join(self.viz_dir, 'precision_recall_curve.png')):
            html_content += """
                        <div class="image-item">
                            <h3>Precision-Recall Curve</h3>
                            <img src="../visualizations/precision_recall_curve.png" alt="Precision-Recall Curve">
                        </div>
            """
        
        # Add similarity distribution
        if os.path.exists(os.path.join(self.viz_dir, 'similarity_distribution.png')):
            html_content += """
                        <div class="image-item">
                            <h3>Similarity Distribution</h3>
                            <img src="../visualizations/similarity_distribution.png" alt="Similarity Distribution">
                        </div>
            """
        
        # Close HTML
        html_content += """
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Training report saved to {html_path} and {json_path}")
    
    def export_model(self, export_format: str = 'keras', quantize: bool = False) -> None:
        """
        Export the trained model in different formats with optional quantization
        
        Args:
            export_format: Format to export ('keras', 'tflite', 'saved_model', 'onnx')
            quantize: Whether to quantize the model for smaller size and faster inference
        """
        if not hasattr(self, 'model') or self.model is None:
            self.logger.error("No model available for export")
            return
        
        export_dir = os.path.join(self.output_dir, 'exported_models')
        os.makedirs(export_dir, exist_ok=True)
        
        if export_format == 'keras':
            # Export as Keras model
            model_path = os.path.join(export_dir, 'face_recognition_model.keras')
            self.model.save_model(model_path)
            self.logger.info(f"Model exported to {model_path}")
            
        elif export_format == 'tflite':
            # Export as TFLite model
            try:
                # Convert to TFLite
                if self.tpu_strategy:
                    with self.tpu_strategy.scope():
                        converter = tf.lite.TFLiteConverter.from_keras_model(self.model.model)
                else:
                    converter = tf.lite.TFLiteConverter.from_keras_model(self.model.model)
                
                if quantize:
                    # Apply quantization for smaller model size
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.uint8
                    converter.inference_output_type = tf.uint8
                    
                    # Representative dataset for quantization calibration
                    def representative_dataset():
                        # Load a small subset of the training data
                        train_images, _ = self.load_dataset(self.train_dir)
                        if len(train_images) > 100:
                            train_images = train_images[:100]
                        for image in train_images:
                            # Add batch dimension
                            image = np.expand_dims(image, axis=0)
                            yield [image.astype(np.float32)]
                    
                    converter.representative_dataset = representative_dataset
                    suffix = "_quantized"
                else:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    suffix = ""
                
                tflite_model = converter.convert()
                
                # Save the model
                tflite_path = os.path.join(export_dir, f'face_recognition_model{suffix}.tflite')
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                    
                self.logger.info(f"TFLite model exported to {tflite_path}")
            except Exception as e:
                self.logger.error(f"Error exporting to TFLite: {str(e)}")
                
        elif export_format == 'saved_model':
            # Export as SavedModel
            try:
                saved_model_path = os.path.join(export_dir, 'saved_model')
                
                if self.tpu_strategy:
                    with self.tpu_strategy.scope():
                        tf.keras.models.save_model(
                            self.model.model,
                            saved_model_path,
                            include_optimizer=False
                        )
                else:
                    tf.keras.models.save_model(
                        self.model.model,
                        saved_model_path,
                        include_optimizer=False
                    )
                    
                self.logger.info(f"SavedModel exported to {saved_model_path}")
                
                # Export signature file with metadata
                signature_path = os.path.join(export_dir, 'model_signature.json')
                signature = {
                    "input_shape": self.model_config.input_shape,
                    "embedding_dim": self.model_config.embedding_dim,
                    "normalize_range": [-1, 1],  # Assuming this is the normalization range used
                    "architecture": "custom_cnn" if self.use_custom_cnn else "resnet50",
                    "use_attention": self.use_attention,
                    "use_arcface": self.use_arcface,
                    "version": "1.0",
                    "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(signature_path, 'w') as f:
                    json.dump(signature, f, indent=4)
                
                self.logger.info(f"Model signature exported to {signature_path}")
                
            except Exception as e:
                self.logger.error(f"Error exporting to SavedModel: {str(e)}")
                
        elif export_format == 'onnx':
            # Export as ONNX model for cross-platform compatibility
            try:
                import tf2onnx
                
                onnx_path = os.path.join(export_dir, 'face_recognition_model.onnx')
                
                # Convert Keras model to ONNX
                if self.tpu_strategy:
                    with self.tpu_strategy.scope():
                        onnx_model, _ = tf2onnx.convert.from_keras(self.model.model)
                else:
                    onnx_model, _ = tf2onnx.convert.from_keras(self.model.model)
                
                # Save ONNX model
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
                self.logger.info(f"ONNX model exported to {onnx_path}")
            except ImportError:
                self.logger.error("tf2onnx package not found. Install with: pip install tf2onnx")
            except Exception as e:
                self.logger.error(f"Error exporting to ONNX: {str(e)}")
                
        else:
            self.logger.error(f"Unsupported export format: {export_format}")
            self.logger.info("Supported formats: 'keras', 'tflite', 'saved_model', 'onnx'")

# Custom callback for dynamic learning rate scheduling with warmup
class DynamicLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr=0.001, min_lr=1e-7, decay_factor=0.7, 
                 patience=5, cooldown=2, warmup_epochs=3, verbose=0):
        """
        Dynamic learning rate scheduler with warmup and cooldown periods
        
        Args:
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            decay_factor: Factor to multiply learning rate when decaying
            patience: Number of epochs with no improvement after which LR will be reduced
            cooldown: Number of epochs to wait before resuming normal operation after LR reduction
            warmup_epochs: Number of epochs for linear warmup
            verbose: Verbosity mode (0: quiet, 1: update messages)
        """
        super(DynamicLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.cooldown = cooldown
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.wait = 0
        self.cooldown_counter = 0
        self.current_lr = initial_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        # Apply warmup during initial epochs
        if epoch < self.warmup_epochs:
            # Linear warmup from initial_lr/10 to initial_lr
            warmup_lr = self.initial_lr / 10 + (self.initial_lr - self.initial_lr / 10) * (epoch / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.lr, warmup_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: Warmup LR set to {warmup_lr:.6f}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get('val_loss')
        
        # Skip LR adjustment during warmup
        if epoch < self.warmup_epochs:
            return
        
        # Get current learning rate
        self.current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        # If in cooldown period, decrease counter and return
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # Check if validation loss improved
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            # If waited for patience epochs with no improvement, reduce LR
            if self.wait >= self.patience:
                # Calculate new learning rate
                new_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                
                # Set new learning rate
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                
                # Reset wait counter and start cooldown
                self.wait = 0
                self.cooldown_counter = self.cooldown
                
                if self.verbose > 0:
                    print(f"\nEpoch {epoch+1}: Reducing LR from {self.current_lr:.6f} to {new_lr:.6f}")
                
                self.current_lr = new_lr

# Custom callback to calculate face recognition accuracy during training
class FaceRecognitionAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, batch_size=32, frequency=5):
        """
        Calculate face recognition accuracy during training
        
        Args:
            validation_data: Tuple of (images, labels) for validation
            batch_size: Batch size for processing
            frequency: Calculate every N epochs to save time
        """
        super(FaceRecognitionAccuracy, self).__init__()
        self.val_images, self.val_labels = validation_data
        self.batch_size = batch_size
        self.frequency = frequency
    
    def on_epoch_end(self, epoch, logs=None):
        # Only calculate every N epochs to save time
        if (epoch + 1) % self.frequency != 0:
            return
        
        # Get embeddings for validation images
        embeddings = []
        for i in range(0, len(self.val_images), self.batch_size):
            batch = self.val_images[i:i+self.batch_size]
            batch_embeddings = self.model.predict(batch, verbose=0)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Calculate cosine similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if i != j:  # Skip self-comparison
                    similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        
        # For each face, find the most similar face and check if labels match
        correct = 0
        total = 0
        
        for i in range(len(self.val_labels)):
            # Get similarities for this sample (excluding self-comparison)
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            
            # Find most similar sample
            most_similar_idx = np.argmax(similarities)
            
            # Check if labels match
            if self.val_labels[i] == self.val_labels[most_similar_idx]:
                correct += 1
            
            total += 1
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0.0
        
        # Add to logs
        logs = logs or {}
        logs['val_face_accuracy'] = accuracy
        
        print(f"\nEpoch {epoch+1}: Face recognition accuracy: {accuracy:.4f}")


def main():
    """Main function to run the training pipeline"""
    # Configuration
    train_dir = "/kaggle/input/vggfacedataset2"  # Default path, can be changed
    val_dir = "/kaggle/input/vggface2/val"
    test_dir = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
    output_dir = "/kaggle/working/"
    
    # Check if directories exist, create sample directories for demonstration if not
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            print(f"Warning: {directory} does not exist. Creating sample directory structure.")
            os.makedirs(directory, exist_ok=True)
            
            # Create sample class directories
            for i in range(3):
                class_dir = os.path.join(directory, f"person_{i}")
                os.makedirs(class_dir, exist_ok=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print system information
    print("\n===== System Information =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Check for TPU
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f"TPU detected: {tpu.cluster_spec().as_dict()}")
        print(f"TPU device: {tpu.master()}")
    except ValueError:
        print("No TPU detected")
    
    print("=============================\n")
    
    # Initialize trainer with TPU support and custom CNN architecture
    trainer = FaceRecognitionTrainer(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=128,  # Larger batch size for TPU
        max_samples_per_class=100,  # Limit samples per class for balancing
        use_gpu=True,
        use_custom_cnn=True,  # Use custom CNN architecture
        use_tpu=True  # Enable TPU usage if available
    )
    
    try:
        # Train model
        trainer.train_model(epochs=100)
        
        # Export model in different formats
        trainer.export_model(export_format='keras')
        trainer.export_model(export_format='tflite')
        
        print("\nTraining pipeline completed successfully!")
        print(f"Results saved to {output_dir}")
    except KeyboardInterrupt:
        print("\nTraining was interrupted by user. Checkpoint has been saved.")
        print("You can resume training by running the script again.")
    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        print("Check the logs for more details.")


if __name__ == "__main__":
    main()