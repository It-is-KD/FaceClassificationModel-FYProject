import os
import cv2
import shutil
import time
import torch
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path
from preprocessingUnit import FacePreprocessor
from model import FaceRecognitionModel, ModelConfig
import zipfile
import logging
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures


class FaceClassifier:
    def __init__(self,
                 model_path: str = None,
                 similarity_threshold: float = 0.75,
                 device: str = 'cuda:0',
                 embedding_dim: int = 512,
                 use_arcface: bool = True):
        """
        Args:
            model_path: Path to pretrained model weights
            similarity_threshold: Threshold for face matching
            device: Device to run MTCNN on ('cpu', 'cuda:0', etc.)
            embedding_dim: Dimension of face embeddings
            use_arcface: Whether to use ArcFace loss
        """

        self._setup_logger() # Initialize logger

        self._check_gpu() # Check for GPU availability

        self.preprocessor = FacePreprocessor( # Initialize preprocessor with enhanced parameters
            target_size=(224, 224),
            use_mtcnn=True,
            normalize_range=(-1, 1),
            device=device,
            confidence_threshold=0.85, #Higher to detech more faces
            margin_percent=0.25
        )

        # Initialize model with enhanced configuration
        config = ModelConfig(
            input_shape=(224, 224, 3),
            embedding_dim=embedding_dim,
            use_pretrained=True,
            base_model_type="resnet50",
            use_attention=True,  # Using attention mechanism
            use_arcface=use_arcface,  # Use ArcFace loss if specified
            dropout_rate=0.5,
            l2_regularization=0.01,
            learning_rate=0.001
        )
        self.model = FaceRecognitionModel(config)

        # Loading pretrained weights
        if model_path and os.path.exists(model_path):
            self.logger.info(f"Loading model weights from {model_path}")
            self.model.load_model(model_path)
        else:
            self.logger.warning("No model weights provided or file not found. Using untrained model.")

        self.similarity_threshold = similarity_threshold

    def _setup_logger(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('FaceClassifier')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _check_gpu(self):
        """Check GPU availability and log information"""
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
                except RuntimeError as e:
                    self.logger.warning(f"Error setting memory growth: {e}")
        else:
            self.logger.warning("No TensorFlow GPU available. Using CPU.")
        
        # Check PyTorch GPU
        if torch.cuda.is_available():
            self.logger.info(f"PyTorch detected {torch.cuda.device_count()} GPU(s):")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"  {i+1}. {torch.cuda.get_device_name(i)}")
        else:
            self.logger.warning("No PyTorch GPU available. Using CPU.")

    def process_reference_image(self, reference_image_path: str) -> np.ndarray:
        """
        Process the reference image and get its embedding with enhanced preprocessing
        """
        self.logger.info(f"Processing reference image: {reference_image_path}")
        
        try:
            # Read image
            image = cv2.imread(reference_image_path)
            if image is None:
                raise ValueError(f"Could not read image: {reference_image_path}")
            
            # Process image to get face embedding
            face = self.preprocessor.process_image(image)
            
            if face is None:
                raise ValueError("No face detected in reference image")
            
            # Generate embedding
            reference_embedding = self.model.get_embedding(face)
            
            self.logger.info(f"Reference embedding generated successfully (shape: {reference_embedding.shape})")
            return reference_embedding
            
        except Exception as e:
            self.logger.error(f"Error processing reference image: {str(e)}")
            raise

    def classify_images(self,
                        input_dir: str,
                        reference_image_path: str,
                        output_dir: str) -> Tuple[List[str], List[str]]:
        """
        Classify images based on similarity to reference image with parallel processing
        """
        start_time = time.time()
        self.logger.info(f"Starting classification for images in {input_dir}")

        # Process reference image
        reference_embedding = self.process_reference_image(reference_image_path)

        # Create output directory structure
        matched_dir = os.path.join(output_dir, 'matched')
        unmatched_dir = os.path.join(output_dir, 'unmatched')
        os.makedirs(matched_dir, exist_ok=True)
        os.makedirs(unmatched_dir, exist_ok=True)

        # Get list of image files
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            self.logger.warning(f"No image files found in {input_dir}")
            return [], []

        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel
        matched_files = []
        unmatched_files = []
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            future_to_filename = {
                executor.submit(self._process_single_image, 
                               os.path.join(input_dir, filename), 
                               reference_embedding): filename 
                for filename in image_files
            }
            
            for future in concurrent.futures.as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    match_found = future.result()
                    input_path = os.path.join(input_dir, filename)
                    
                    if match_found:
                        shutil.copy2(input_path, matched_dir)
                        matched_files.append(filename)
                        self.logger.info(f"Match found: {filename}")
                    else:
                        shutil.copy2(input_path, unmatched_dir)
                        unmatched_files.append(filename)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {str(e)}")
                    # Copy to unmatched in case of error
                    try:
                        shutil.copy2(os.path.join(input_dir, filename), unmatched_dir)
                    except Exception:
                        pass
                    unmatched_files.append(filename)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Classification completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Matched: {len(matched_files)}, Unmatched: {len(unmatched_files)}")

        return matched_files, unmatched_files

    def _process_single_image(self, image_path: str, reference_embedding: np.ndarray) -> bool:
        """
        Process a single image and determine if it matches the reference
        
        Args:
            image_path: Path to the image
            reference_embedding: Embedding of the reference face
            
        Returns:
            True if match found, False otherwise
        """
        try:
            # Use robust image loading instead of simple cv2.imread
            image = self._robust_image_load(image_path)
            
            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return False
                
            detections = self.preprocessor.detect_faces(image) # Detect faces
            
            if not detections:
                return False
                
            for detection in detections: # Check each detected face for a match

                face = self.preprocessor.preprocess_face(image, detection)
                
                embedding = self.model.get_embedding(face)
                
                similarity = cosine_similarity(
                    reference_embedding,
                    embedding
                )[0][0]
                
                if similarity >= self.similarity_threshold:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error in _process_single_image: {str(e)}")
            return False

    def _robust_image_load(self, image_path):
        """
        Robust image loading function that tries multiple methods
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array in BGR format (OpenCV format) or None if failed
        """
        # First try: standard OpenCV loading
        image = cv2.imread(image_path)
        if image is not None:
            return image
        
        self.logger.warning(f"OpenCV failed to read image: {image_path}, trying alternative methods...")
        
        # Check if file exists
        if not os.path.exists(image_path):
            self.logger.error(f"Error: File does not exist: {image_path}")
            return None
            
        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            self.logger.error(f"Error: File is empty (0 bytes): {image_path}")
            return None
        
        # Try PIL/Pillow
        try:
            from PIL import Image, UnidentifiedImageError
            try:
                pil_image = Image.open(image_path)
                # Get image format for diagnostics
                img_format = pil_image.format
                # Convert to RGB (PIL uses RGB, OpenCV uses BGR)
                pil_image = pil_image.convert('RGB')
                # Convert PIL image to numpy array
                image = np.array(pil_image)
                # Convert RGB to BGR for OpenCV processing
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.logger.info(f"Successfully loaded image using PIL: {image_path} (Format: {img_format})")
                return image
            except UnidentifiedImageError:
                self.logger.error(f"PIL Error: Unidentified image format in file: {image_path}")
                return None
            except Exception as pil_error:
                self.logger.error(f"PIL Error: {str(pil_error)} when reading: {image_path}")
                
                # Try imageio as a last resort
                try:
                    import imageio
                    image = imageio.imread(image_path)
                    # Convert to BGR for OpenCV if needed
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    self.logger.info(f"Successfully loaded image using imageio: {image_path}")
                    return image
                except ImportError:
                    self.logger.error("imageio library not available for fallback image loading")
                    return None
                except Exception as imageio_error:
                    self.logger.error(f"All image loading methods failed for: {image_path}")
                    self.logger.error(f"Detailed error: {str(imageio_error)}")
                    return None
        except ImportError:
            self.logger.error("PIL/Pillow library not available for fallback image loading")
            return None
        
        return None

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
                        
        self.logger.info(f"Zip file created: {zip_path}")

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
            self.logger.info(f"Visualization saved to {save_dir}")


def main():
    input_dir = "../inputCluster/"
    reference_image = "../RAJ_4024.JPG"
    output_dir = "../"
    output_zip = "classified_images.zip"
    model_path = "path/to/model/weights.h5"
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    try:
        # Print system info
        print("\n===== System Information =====")
        print(f"PyTorch version: {torch.__version__}")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print("=============================\n")
        
        # Initialize classifier with enhanced parameters
        classifier = FaceClassifier(
            model_path=model_path, 
            device=device,
            similarity_threshold=0.85,  # Adjusted threshold
            embedding_dim=512,  # Larger embedding dimension
            use_arcface=True  # Use ArcFace for better accuracy
        )
        
        # Process reference image with visualization
        f_classifier = FaceClassifier(device=device)
        f_classifier.process_single_image(reference_image, visualize=True)

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
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()