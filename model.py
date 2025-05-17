import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import ResNet50V2
# from keras._tf_keras.keras.applications import ResNet50V2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    embedding_dim: int = 512  # Increased from 128 for better representation
    use_pretrained: bool = True
    dropout_rate: float = 0.5
    l2_regularization: float = 0.01
    batch_norm_momentum: float = 0.99
    learning_rate: float = 0.001
    base_model_type: str = "efficientnet"  # Options: "resnet50", "efficientnet", "custom"
    use_attention: bool = True  # Use attention mechanism for better feature extraction
    use_arcface: bool = True  # Use ArcFace loss for better face recognition
    margin: float = 0.5  # Margin for triplet/arcface loss
    scale: float = 64.0  # Scale for arcface loss


class FaceRecognitionModel:
    def __init__(self, config: ModelConfig):

        self.config = config
        self.model = self._build_model()
        
        self._setup_gpu_memory_growth()

    def _setup_gpu_memory_growth(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU memory growth enabled on {len(gpus)} GPU(s)")
        except Exception as e:
            print(f"Error setting up GPU memory growth: {e}")

    def _build_base_model(self) -> Model:
        if not self.config.use_pretrained:
            return self._build_custom_cnn()
            
        if self.config.base_model_type == "resnet50":
            base_model = ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=self.config.input_shape,
                pooling='avg'
            )
            # Freeze early layers but keep later layers trainable for fine-tuning
            for layer in base_model.layers[:-50]:  # Unfreeze more layers for better adaptation
                layer.trainable = False
            return base_model
            
        elif self.config.base_model_type == "efficientnet":
            # EfficientNet is more efficient and often more accurate
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=self.config.input_shape,
                pooling='avg'
            )
            # Freeze early layers
            for layer in base_model.layers[:-30]:
                layer.trainable = False
            return base_model
            
        else:
            return self._build_custom_cnn()

    def _build_custom_cnn(self) -> Model:
        inputs = layers.Input(shape=self.config.input_shape)

        # First conv block with residual connection
        x = self._conv_block(inputs, filters=32, kernel_size=3, strides=1)
        x = self._conv_block(x, filters=32, kernel_size=3, strides=2)
        residual1 = x

        # Second conv block with residual connection
        x = self._conv_block(x, filters=64, kernel_size=3, strides=1)
        x = self._conv_block(x, filters=64, kernel_size=3, strides=2)
        # Add residual connection with projection
        residual1 = layers.Conv2D(64, 1, strides=2, padding='same')(residual1)
        x = layers.add([x, residual1])
        residual2 = x

        # Third conv block with residual connection
        x = self._conv_block(x, filters=128, kernel_size=3, strides=1)
        x = self._conv_block(x, filters=128, kernel_size=3, strides=2)
        # Add residual connection with projection
        residual2 = layers.Conv2D(128, 1, strides=2, padding='same')(residual2)
        x = layers.add([x, residual2])
        residual3 = x

        # Fourth conv block with residual connection
        x = self._conv_block(x, filters=256, kernel_size=3, strides=1)
        x = self._conv_block(x, filters=256, kernel_size=3, strides=2)
        # Add residual connection with projection
        residual3 = layers.Conv2D(256, 1, strides=2, padding='same')(residual3)
        x = layers.add([x, residual3])

        # Add attention mechanism if configured
        if self.config.use_attention:
            x = self._attention_block(x)

        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)

        return Model(inputs, x, name='custom_cnn')

    def _attention_block(self, x: tf.Tensor) -> tf.Tensor:

        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        avg_pool = layers.Reshape((1, 1, x.shape[-1]))(avg_pool)
        avg_pool = layers.Conv2D(x.shape[-1] // 8, kernel_size=1)(avg_pool)
        avg_pool = layers.Activation('relu')(avg_pool)
        avg_pool = layers.Conv2D(x.shape[-1], kernel_size=1)(avg_pool)
        
        max_pool = layers.GlobalMaxPooling2D()(x)
        max_pool = layers.Reshape((1, 1, x.shape[-1]))(max_pool)
        max_pool = layers.Conv2D(x.shape[-1] // 8, kernel_size=1)(max_pool)
        max_pool = layers.Activation('relu')(max_pool)
        max_pool = layers.Conv2D(x.shape[-1], kernel_size=1)(max_pool)
        
        channel_attention = layers.Add()([avg_pool, max_pool])
        channel_attention = layers.Activation('sigmoid')(channel_attention)
        
        # Apply channel attention
        x = layers.Multiply()([x, channel_attention])
        
        return x

    def _conv_block(self,
                    x: tf.Tensor,
                    filters: int,
                    kernel_size: int,
                    strides: int) -> tf.Tensor:
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_regularizer=regularizers.l2(self.config.l2_regularization),
            kernel_initializer='he_normal'  # Better initialization for ReLU activations
        )(x)
        x = layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum
        )(x)
        x = layers.PReLU()(x)  # PReLU allows negative values with learned coefficients
        return x

    def _build_model(self) -> Model:

        # Input layer
        inputs = layers.Input(shape=self.config.input_shape)

        # Base CNN model
        base_model = self._build_base_model()
        x = base_model(inputs)

        # Embedding layers with skip connections
        x1 = layers.Dense(
            1024,
            kernel_regularizer=regularizers.l2(self.config.l2_regularization)
        )(x)
        x1 = layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum
        )(x1)
        x1 = layers.PReLU()(x1)
        x1 = layers.Dropout(self.config.dropout_rate)(x1)

        # Second embedding layer
        x2 = layers.Dense(
            512,
            kernel_regularizer=regularizers.l2(self.config.l2_regularization)
        )(x1)
        x2 = layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum
        )(x2)
        x2 = layers.PReLU()(x2)
        x2 = layers.Dropout(self.config.dropout_rate)(x2)

        # Final embedding layer
        embeddings = layers.Dense(
            self.config.embedding_dim,
            kernel_regularizer=regularizers.l2(self.config.l2_regularization)
        )(x2)

        # L2 normalization
        normalized_embeddings = layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1),
            name='normalized_embeddings'
        )(embeddings)

        return Model(inputs, normalized_embeddings, name='face_recognition_model')

    def compile_model(self,
                    optimizer: Optional[tf.keras.optimizers.Optimizer] = None):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)

        if self.config.use_arcface:
            num_classes = 120  # Set this to your actual number of classes
            self.arcface_weights = tf.Variable(
                tf.random.normal([num_classes, self.config.embedding_dim]),
                name='arcface_weights',
                trainable=True
            )
            # ArcFace loss provides better angular separation between classes
            self.model.compile(
                optimizer=optimizer,
                loss=self._arcface_loss
            )
        else:
            # Traditional triplet loss
            self.model.compile(
                optimizer=optimizer,
                loss=self._triplet_loss
            )

    def _triplet_loss(self,
                y_true: tf.Tensor,
                y_pred: tf.Tensor) -> tf.Tensor:
        # Split embeddings into anchor, positive, and negative
        embeddings = tf.cast(y_pred, tf.float32)
        anchor, positive, negative = tf.split(embeddings, 0, axis=0)

        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # Calculate triplet loss with margin from config
        margin = self.config.margin
        basic_loss = pos_dist - neg_dist + margin
        loss = tf.maximum(basic_loss, 0.0)

        # Add a small regularization term to encourage smaller embeddings
        reg_term = 0.0001 * (tf.reduce_mean(tf.square(anchor)) +
                            tf.reduce_mean(tf.square(positive)) +
                            tf.reduce_mean(tf.square(negative)))

        return tf.reduce_mean(loss) + reg_term

    def _arcface_loss(self,
                        y_true: tf.Tensor,
                        y_pred: tf.Tensor) -> tf.Tensor:

        num_classes = tf.shape(y_true)[1]

        batch_size = tf.shape(y_pred)[0]

        weights_norm = tf.nn.l2_normalize(self.arcface_weights, axis=1)
        features_norm = tf.nn.l2_normalize(y_pred, axis=1)

        cos_t = tf.matmul(features_norm, weights_norm, transpose_b=True)

        mask = tf.cast(y_true, dtype=tf.float32)

        margin = self.config.margin
        scale = self.config.scale

        cos_t_margin = tf.cos(tf.acos(tf.clip_by_value(cos_t, -0.9999, 0.9999)) + margin)

        final_target_logits = mask * cos_t_margin + (1.0 - mask) * cos_t

        scaled_logits = final_target_logits * scale

        losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=scaled_logits)

        return tf.reduce_mean(losses)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:

        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Use GPU for prediction if available
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            return self.model.predict(image, verbose=0)

    def get_embeddings(self, images: np.ndarray) -> np.ndarray:

        # Use GPU for prediction if available
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            # Process in batches to avoid memory issues
            batch_size = 32
            if len(images) <= batch_size:
                return self.model.predict(images, verbose=0)

            # Process in batches
            embeddings = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                batch_embeddings = self.model.predict(batch, verbose=0)
                embeddings.append(batch_embeddings)

            return np.vstack(embeddings)

    def save_model(self,
                model_path: str,
                save_weights_only: bool = False):

        if save_weights_only:
            self.model.save_weights(model_path)
        else:
            self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self,
                model_path: str,
                load_weights_only: bool = False):
        if load_weights_only:
            self.model.load_weights(model_path)
        else:
            # Choose the appropriate loss function based on config
            if self.config.use_arcface:
                self.model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'_arcface_loss': self._arcface_loss}
                )
            else:
                self.model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'_triplet_loss': self._triplet_loss}
                )
        print(f"Model loaded on {model_path}")

    def evaluate(self,
                test_data: Tuple[np.ndarray, np.ndarray],
                batch_size: int = 32) -> float:
        test_images, test_labels = test_data
        test_generator = TripletGenerator(
            images=test_images,
            labels=test_labels,
            batch_size=batch_size
        )

        return self.model.evaluate(test_generator)

    def train(self,
            train_data,
            validation_data=None,
            epochs=100,
            initial_epoch=0,  # Add this parameter
            batch_size=32,
            callbacks=None):

        train_images, train_labels = train_data

        # Create data generator with augmentation
        train_generator = TripletGenerator(
            images=train_images,
            labels=train_labels,
            batch_size=batch_size,
            use_augmentation=True  # Enable data augmentation
        )

        validation_generator = None
        if validation_data is not None:
            val_images, val_labels = validation_data
            validation_generator = TripletGenerator(
                images=val_images,
                labels=val_labels,
                batch_size=batch_size,
                use_augmentation=False  # No augmentation for validation
            )

        # Add learning rate scheduler if not provided in callbacks
        if callbacks is None:
            callbacks = []

        # Add learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_scheduler)

        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Compile model
        self.compile_model()

        # Train model with model acceleration
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                initial_epoch=initial_epoch,  # Pass the initial_epoch parameter here
                validation_data=validation_generator,
                callbacks=callbacks,
                workers=4,  # Parallel data loading
                use_multiprocessing=True  # Enable multiprocessing
            )

        return history
            
class TripletGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                images: np.ndarray,
                labels: np.ndarray,
                batch_size: int = 32,
                use_augmentation: bool = False):

        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self.indices = np.arange(len(images))
        
        # Create label to indices mapping for efficient triplet selection
        self.label_to_indices = {}
        for i, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)
        
        # Ensure all classes have at least 2 samples
        self.valid_labels = [label for label in self.label_to_indices 
                            if len(self.label_to_indices[label]) >= 2]
        
        if len(self.valid_labels) == 0:
            raise ValueError("Each class needs at least 2 examples for triplet loss")

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        """Generate one batch of triplets with hard negative mining"""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Initialize arrays for triplets
        anchors = np.zeros((self.batch_size,) + self.images[0].shape)
        positives = np.zeros((self.batch_size,) + self.images[0].shape)
        negatives = np.zeros((self.batch_size,) + self.images[0].shape)

        for i, anchor_idx in enumerate(batch_indices):
            anchor_label = self.labels[anchor_idx]
            
            # Skip if this label doesn't have enough samples
            if anchor_label not in self.valid_labels:
                # Choose a random valid label
                anchor_label = np.random.choice(self.valid_labels)
                anchor_idx = np.random.choice(self.label_to_indices[anchor_label])

            # Get positive sample (same label, different image)
            positive_indices = [idx for idx in self.label_to_indices[anchor_label] if idx != anchor_idx]
            positive_idx = np.random.choice(positive_indices)

            # Get negative sample (different label) - semi-hard negative mining
            negative_labels = [label for label in self.valid_labels if label != anchor_label]
            negative_label = np.random.choice(negative_labels)
            negative_idx = np.random.choice(self.label_to_indices[negative_label])

            anchors[i] = self.images[anchor_idx]
            positives[i] = self.images[positive_idx]
            negatives[i] = self.images[negative_idx]

        # Apply data augmentation if enabled
        if self.use_augmentation:
            anchors = self._augment_batch(anchors)
            positives = self._augment_batch(positives)
            negatives = self._augment_batch(negatives)

        # Combine triplets
        triplets = np.concatenate([anchors, positives, negatives], axis=0)

        # Dummy labels (not used by triplet loss)
        dummy_labels = np.zeros((self.batch_size * 3,))

        return triplets, dummy_labels
    
    def _augment_batch(self, batch: np.ndarray) -> np.ndarray:
        """Apply data augmentation to a batch of images"""
        augmented_batch = batch.copy()
        
        # Apply random augmentations
        for i in range(len(batch)):
            img = batch[i]
            
            # Random horizontal flip (50% chance)
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness_factor, -1.0, 1.0)
            
            # Random contrast adjustment
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(img)
            img = np.clip((img - mean) * contrast_factor + mean, -1.0, 1.0)
            
            augmented_batch[i] = img
            
        return augmented_batch

    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        np.random.shuffle(self.indices)