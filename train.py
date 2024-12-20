# import numpy as np
# from model import ModelConfig, FaceRecognitionModel
# from preprocessingUnit import FacePreprocessor

# # Initialize preprocessor
# preprocessor = FacePreprocessor(target_size=(224, 224))

# # Load and preprocess your dataset
# # Assuming you have a directory structure with face images organized by identity
# def load_dataset(data_dir):
#     # Load images and labels
#     # Return preprocessed images and corresponding identity labels
#     pass

# # Load training data
# train_images, train_labels = load_dataset('path/to/train/data')
# val_images, val_labels = load_dataset('path/to/val/data')
# test_images, test_labels = load_dataset('path/to/test/data')

# # Initialize model
# config = ModelConfig(
#     input_shape=(224, 224, 3),
#     embedding_dim=128,
#     use_pretrained=False,
#     dropout_rate=0.5,
#     l2_regularization=0.01,
#     batch_norm_momentum=0.99,
#     learning_rate=0.001
# )

# model = FaceRecognitionModel(config)

# # Train the model
# history = model.train(
#     train_data=(train_images, train_labels),
#     validation_data=(val_images, val_labels),
#     epochs=100,
#     batch_size=32
# )

# # Evaluate model
# test_loss = model.evaluate(
#     test_data=(test_images, test_labels)
# )

# # Save the trained model
# model.save_model('trained_face_recognition_model.h5')model = FaceRecognitionModel(config)

import os
import numpy as np
import matplotlib.pyplot as plt
from model import ModelConfig, FaceRecognitionModel
from preprocessingUnit import FacePreprocessor
import tensorflow as tf

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TrainingPipeline')

# Initialize preprocessor
preprocessor = FacePreprocessor(target_size=(224, 224))

def load_dataset(data_dir):
    images = []
    labels = []
    logger.info(f"Loading dataset from: {data_dir}")
    
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        return np.array(images), np.array(labels)
        
    subdirs = os.listdir(data_dir)
    logger.info(f"Found {len(subdirs)} subdirectories")
    
    # for person_id, person_dir in enumerate(subdirs):
    #     person_path = os.path.join(data_dir, person_dir)
    #     if os.path.isdir(person_path):
    #         image_files = [f for f in os.listdir(person_path) 
    #                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #         logger.info(f"Processing {person_dir}: {len(image_files)} images")
            
    #         for image_file in image_files:
    #             image_path = os.path.join(person_path, image_file)
    #             face = preprocessor.preprocess_image(image_path)
    #             images.append(face)
    #             labels.append(person_id)
    
    # return np.array(images), np.array(labels)

    for person_id, person_dir in enumerate(os.listdir(data_dir)):
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, image_file)
                    face = preprocessor.preprocess_image(image_path)
                    if face is not None and face.shape == (224, 224, 3):
                        images.append(face)
                        labels.append(person_id)
    
    logger.info(f"Loaded {len(images)} images with {len(set(labels))} unique identities")
    return np.stack(images), np.array(labels)

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training history plot saved to {save_path}")

def main():
    # Data paths
    # train_dir = "path/to/vggface2_train"
    # val_dir = "path/to/celeba"
    # test_dir = "path/to/lfw"

    train_dir = "./VGG-Face2/data/vggface2_train/train"
    val_dir = "./CelebsA/img_align_celeba"
    test_dir = "./LabbelledFacesInTheWild/lfw-deepfunneled/lfw-deepfunneled"
    
    # Create output directories
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Load datasets
    train_images, train_labels = load_dataset(train_dir)
    val_images, val_labels = load_dataset(val_dir)
    test_images, test_labels = load_dataset(test_dir)
    
    # Initialize model
    config = ModelConfig(
        input_shape=(224, 224, 3),
        embedding_dim=128,
        use_pretrained=False,
        dropout_rate=0.5,
        l2_regularization=0.01,
        batch_norm_momentum=0.99,
        learning_rate=0.001
    )
    
    model = FaceRecognitionModel(config)
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'model_checkpoints/best_model.keras',  # Changed from .h5 to .keras
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            monitor='val_loss',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # Train the model
    logger.info("Starting model training...")
    history = model.train(  
        train_data=(train_images, train_labels),
        validation_data=(val_images, val_labels),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    logger.info("Evaluating model on test set...")
    test_loss = model.evaluate(
        test_data=(test_images, test_labels)
    )
    logger.info(f"Test Loss: {test_loss}")
    
    # Save final model
    final_model_path = 'trained_face_recognition_model.keras'
    model.save_model(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    main()