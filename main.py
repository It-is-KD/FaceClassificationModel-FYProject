# main.py

import os
import cv2
import shutil
from typing import List, Tuple
import numpy as np
from pathlib import Path
from preprocessingUnit import FacePreprocessor
from model import FaceRecognitionModel, ModelConfig
import zipfile
import logging
from sklearn.metrics.pairwise import cosine_similarity


class FaceClassifier:
    def __init__(self,
                 model_path: str = None,
                 similarity_threshold: float = 0.85):
        """
        Initialize the face classifier

        Args:
            model_path: Path to pretrained model weights
            similarity_threshold: Threshold for face matching
        """
        # Initialize logger
        self._setup_logger()

        # Initialize preprocessor
        self.preprocessor = FacePreprocessor(
            target_size=(224, 224),
            use_mtcnn=True,
            normalize_range=(-1, 1)
        )

        # Initialize model
        config = ModelConfig(
            input_shape=(224, 224, 3),
            embedding_dim=128,
            use_pretrained=True
        )
        self.model = FaceRecognitionModel(config)

        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            self.logger.info(f"Loading model weights from {model_path}")
            self.model.load_model(model_path)

        self.similarity_threshold = similarity_threshold

    def _setup_logger(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('FaceClassifier')
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def process_reference_image(self, reference_image_path: str) -> np.ndarray:
        """
        Process the reference image and get its embedding
        """
        self.logger.info(f"Processing reference image: {reference_image_path}")
        
        try:
            # Detect faces in reference image
            image = cv2.imread(reference_image_path)
            detections = self.preprocessor.detect_faces(image)
            
            if not detections:
                raise ValueError("No face detected in reference image")
                
            # Get the first detected face
            face_data = detections[0]
            x, y, w, h = face_data['bbox']
            
            # Extract and preprocess the face
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, self.preprocessor.target_size)
            
            # Normalize the face image
            face = (face - 127.5) / 127.5
            face = np.expand_dims(face, axis=0)
            
            # Generate embedding
            reference_embedding = self.model.get_embedding(face)
            
            return reference_embedding
            
        except Exception as e:
            self.logger.error(f"Error processing reference image: {str(e)}")
            raise
            
    def classify_images(self,
                        input_dir: str,
                        reference_image_path: str,
                        output_dir: str) -> Tuple[List[str], List[str]]:
        """
        Classify images based on similarity to reference image
        """
        self.logger.info(f"Starting classification for images in {input_dir}")

        # Process reference image
        reference_embedding = self.process_reference_image(reference_image_path)

        # Create output directory structure
        matched_dir = os.path.join(output_dir, 'matched')
        unmatched_dir = os.path.join(output_dir, 'unmatched')
        os.makedirs(matched_dir, exist_ok=True)
        os.makedirs(unmatched_dir, exist_ok=True)

        matched_files = []
        unmatched_files = []

        # Process each image in input directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)

                try:
                    # Process image and get faces
                    processed_faces = self.preprocessor.preprocess_image(
                        input_path,
                        return_all_faces=True
                    )

                    # Check if any face matches
                    match_found = False
                    for face in processed_faces:
                        # Generate embedding
                        embedding = self.model.get_embedding(face)

                        # Calculate similarity
                        similarity = cosine_similarity(
                            reference_embedding.reshape(1, -1),
                            embedding.reshape(1, -1)
                        )[0][0]

                        if similarity >= self.similarity_threshold:
                            match_found = True
                            break

                    # Copy file to appropriate directory
                    
                    if match_found:
                        shutil.copy2(input_path, unmatched_dir)  # Changed to unmatched_dir
                        unmatched_files.append(filename)  # Changed to unmatched_files
                        self.logger.info(f"Match found: {filename}")
                    else:
                        shutil.copy2(input_path, matched_dir)  # Changed to matched_dir
                        matched_files.append(filename)  # Changed to matched_files

                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {str(e)}")
                    unmatched_files.append(filename)

        return matched_files, unmatched_files

    def create_output_zip(self,
                          output_dir: str,
                          zip_path: str):
        """
        Create a zip file of the classified images

        Args:
            output_dir: Directory containing classified images
            zip_path: Path where zip file should be created
        """
        self.logger.info(f"Creating zip file: {zip_path}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add matched images
            matched_dir = os.path.join(output_dir, 'matched')
            if os.path.exists(matched_dir):
                for root, _, files in os.walk(matched_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('matched', file)
                        zipf.write(file_path, arcname)

            # Add unmatched images
            unmatched_dir = os.path.join(output_dir, 'unmatched')
            if os.path.exists(unmatched_dir):
                for root, _, files in os.walk(unmatched_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('unmatched', file)
                        zipf.write(file_path, arcname)
    def process_single_image(self, image_path: str, visualize: bool = False) -> None:
        """
        Process a single image and visualize preprocessing steps
        
        Args:
            image_path: Path to the input image
            visualize: If True, save visualization of preprocessing steps
        """
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return
            
        if visualize:
            save_dir = os.path.join("preprocessing_visualization", 
                                os.path.splitext(os.path.basename(image_path))[0])
            self.preprocessor.visualize_preprocessing(image, save_dir)



def main():
    # Configuration
    input_dir = "./path to input dir"
    reference_image = "./path to the ref img to be used to classify"
    output_dir = "./"
    output_zip = "classified_images.zip"
    model_path = "path/to/model/weights.h5" 

    try:
        # Initialize classifier
        classifier = FaceClassifier(model_path)
        f_classifier = FaceClassifier()
        f_classifier.process_single_image("./Intel Unison/IMG_7640.png", visualize=True)

        # Perform classification
        matched, unmatched = classifier.classify_images(
            input_dir,
            reference_image,
            output_dir
        )

        # Create zip file
        classifier.create_output_zip(output_dir, output_zip)

        # Print summary
        print(f"\nClassification Summary:")
        print(f"Total matched images: {len(matched)}")
        print(f"Total unmatched images: {len(unmatched)}")
        print(f"\nResults have been saved to {output_zip}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
