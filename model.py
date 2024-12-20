import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
# from tensorflow.keras.applications import ResNet50V2
from keras._tf_keras.keras.applications import ResNet50V2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the face recognition model"""
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    embedding_dim: int = 128
    use_pretrained: bool = True
    dropout_rate: float = 0.5
    l2_regularization: float = 0.01
    batch_norm_momentum: float = 0.99
    learning_rate: float = 0.001

class FaceRecognitionModel:
    def __init__(self, config: ModelConfig):
        """
        Initialize the face recognition model

        Args:
            config: ModelConfig object containing model parameters
        """
        self.config = config
        self.model = self._build_model()

    def _build_base_model(self) -> Model:
        """
        Build the base CNN model (either pretrained or custom)
        """
        if self.config.use_pretrained:
            base_model = ResNet50V2(
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
        """
        Build a custom CNN architecture
        """
        inputs = layers.Input(shape=self.config.input_shape)

        # First conv block
        x = self._conv_block(inputs, filters=32, kernel_size=3, strides=1)
        x = self._conv_block(x, filters=32, kernel_size=3, strides=2)

        # Second conv block
        x = self._conv_block(x, filters=64, kernel_size=3, strides=1)
        x = self._conv_block(x, filters=64, kernel_size=3, strides=2)

        # Third conv block
        x = self._conv_block(x, filters=128, kernel_size=3, strides=1)
        x = self._conv_block(x, filters=128, kernel_size=3, strides=2)

        # Fourth conv block
        x = self._conv_block(x, filters=256, kernel_size=3, strides=1)
        x = self._conv_block(x, filters=256, kernel_size=3, strides=2)

        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)

        return Model(inputs, x, name='custom_cnn')

    def _conv_block(self,
                    x: tf.Tensor,
                    filters: int,
                    kernel_size: int,
                    strides: int) -> tf.Tensor:
        """
        Create a convolution block with batch normalization and activation
        """
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_regularizer=regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum
        )(x)
        x = layers.PReLU()(x)
        return x

    def _build_model(self) -> Model:
        """
        Build the complete face recognition model
        """
        # Input layer
        inputs = layers.Input(shape=self.config.input_shape)

        # Base CNN model
        base_model = self._build_base_model()
        x = base_model(inputs)

        # Embedding layers
        x = layers.Dense(
            512,
            kernel_regularizer=regularizers.l2(self.config.l2_regularization)
        )(x)
        x = layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum
        )(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        # Final embedding layer
        embeddings = layers.Dense(
            self.config.embedding_dim,
            kernel_regularizer=regularizers.l2(self.config.l2_regularization)
        )(x)

        # L2 normalization
        normalized_embeddings = layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1),
            name='normalized_embeddings'
        )(embeddings)

        return Model(inputs, normalized_embeddings, name='face_recognition_model')

    def compile_model(self,
                      optimizer: Optional[tf.keras.optimizers.Optimizer] = None):
        """
        Compile the model with triplet loss
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self._triplet_loss
        )

    def _triplet_loss(self,
                      y_true: tf.Tensor,
                      y_pred: tf.Tensor,
                      margin: float = 0.2) -> tf.Tensor:
        """
        Implementation of triplet loss

        Args:
            y_true: Unused (required by Keras API)
            y_pred: Tensor containing anchor, positive, and negative embeddings
            margin: Minimum distance between positive and negative pairs

        Returns:
            Triplet loss value
        """
        # Split embeddings into anchor, positive, and negative
        embeddings = tf.cast(y_pred, tf.float32)
        anchor, positive, negative = tf.split(embeddings, 3, axis=0)

        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # Calculate triplet loss
        basic_loss = pos_dist - neg_dist + margin
        loss = tf.maximum(basic_loss, 0.0)

        return tf.reduce_mean(loss)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Generate embedding for a single face image

        Args:
            image: Preprocessed face image

        Returns:
            Face embedding vector
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        return self.model.predict(image)

    def get_embeddings(self, images: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for multiple face images

        Args:
            images: Batch of preprocessed face images

        Returns:
            Array of face embedding vectors
        """
        return self.model.predict(images)

    def save_model(self,
                   model_path: str,
                   save_weights_only: bool = False):
        """
        Save the model to disk
        """
        if save_weights_only:
            self.model.save_weights(model_path)
        else:
            self.model.save(model_path)

    def load_model(self,
                   model_path: str,
                   load_weights_only: bool = False):
        """
        Load the model from disk
        """
        if load_weights_only:
            self.model.load_weights(model_path)
        else:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'_triplet_loss': self._triplet_loss}
            )

    def evaluate(self, 
                test_data: Tuple[np.ndarray, np.ndarray],
                batch_size: int = 32) -> float:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Tuple of (images, labels) for testing
            batch_size: Batch size for evaluation
            
        Returns:
            Test loss value
        """
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
            batch_size=32,
            callbacks=None):
        """
        Train the face recognition model
        
        Args:
            train_data: Tuple of (images, labels) for training
            validation_data: Optional tuple of (images, labels) for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
        """
        train_images, train_labels = train_data
        train_generator = TripletGenerator(
            images=train_images,
            labels=train_labels,
            batch_size=batch_size
        )
        
        validation_generator = None
        if validation_data is not None:
            val_images, val_labels = validation_data
            validation_generator = TripletGenerator(
                images=val_images,
                labels=val_labels,
                batch_size=batch_size
            )
        
        self.compile_model()
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        return history
        
class TripletGenerator(tf.keras.utils.Sequence):
    """Data generator for triplet loss training"""

    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 batch_size: int = 32):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.arange(len(images))

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        """Generate one batch of triplets"""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Initialize arrays for triplets
        anchors = np.zeros((self.batch_size,) + self.images[0].shape)
        positives = np.zeros((self.batch_size,) + self.images[0].shape)
        negatives = np.zeros((self.batch_size,) + self.images[0].shape)

        for i, anchor_idx in enumerate(batch_indices):
            anchor_label = self.labels[anchor_idx]

            # Get positive sample (same label)
            positive_indices = np.where(self.labels == anchor_label)[0]
            positive_idx = np.random.choice(positive_indices)

            # Get negative sample (different label)
            negative_indices = np.where(self.labels != anchor_label)[0]
            negative_idx = np.random.choice(negative_indices)

            anchors[i] = self.images[anchor_idx]
            positives[i] = self.images[positive_idx]
            negatives[i] = self.images[negative_idx]

        # Combine triplets
        triplets = np.concatenate([anchors, positives, negatives], axis=0)

        # Dummy labels (not used by triplet loss)
        dummy_labels = np.zeros((self.batch_size * 3,))

        return triplets, dummy_labels