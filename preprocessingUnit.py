import cv2
import numpy as np
import os
import torch
from facenet_pytorch import MTCNN
from typing import Tuple, List, Dict, Union, Optional
from dataclasses import dataclass
from pathlib import Path
import math
from PIL import Image


@dataclass
class ProcessedFace:
    """Data class to store processed face information"""
    face_image: np.ndarray
    original_filename: str
    confidence: float
    face_id: int  # To distinguish multiple faces from same image
    bbox: Tuple[int, int, int, int]
    quality_score: float = 0.0  # Added quality score field


class FacePreprocessor:
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224), 
                 use_mtcnn: bool = True, 
                 normalize_range: Tuple[float, float] = (-1, 1),  # Changed default to (-1, 1) for better model performance
                 device: str = 'cuda:0',
                 confidence_threshold: float = 0.85,  # Slightly lower threshold to detect more faces
                 margin_percent: float = 0.25,  # Increased margin for better face alignment
                 quality_threshold: float = 0.5):  # Added quality threshold parameter
        """
        Initialize the face preprocessing pipeline with enhanced MTCNN

        Args:
            target_size: Output size for face images
            use_mtcnn: If True, use MTCNN for face detection/alignment
            normalize_range: Range for pixel normalization
            device: Device to run MTCNN on ('cpu', 'cuda:0', etc.)
            confidence_threshold: Minimum confidence threshold for face detection
            margin_percent: Percentage of face size to add as margin
            quality_threshold: Minimum quality score for face acceptance
        """
        self.target_size = target_size
        self.normalize_range = normalize_range
        self.confidence_threshold = confidence_threshold
        self.margin_percent = margin_percent
        self.quality_threshold = quality_threshold

        # Set device based on availability
        if torch.cuda.is_available() and 'cuda' in device:
            self.device = device
            print(f"Using GPU device: {device}")
        else:
            self.device = 'cpu'
            print("CUDA not available, using CPU")

        # Initialize MTCNN face detector with optimized parameters
        if use_mtcnn:
            try:
                self.face_detector = MTCNN(
                    image_size=target_size[0],
                    margin=int(target_size[0] * margin_percent),  # Dynamic margin based on image size
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.8],  # Adjusted thresholds for better detection
                    factor=0.709,  # Scale factor for image pyramid
                    post_process=True,
                    keep_all=True,  # Keep all detected faces
                    device=self.device,
                    select_largest=False  # Don't just select largest face
                )
                print(f"Enhanced MTCNN initialized on device: {self.device}")
            except Exception as e:
                print(f"Error initializing MTCNN: {e}")
                raise RuntimeError(f"Failed to initialize MTCNN: {e}")
        else:
            raise ValueError("Only MTCNN is supported in this version")

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image and return their bounding boxes and landmarks

        Args:
            image: BGR image from OpenCV

        Returns:
            List of dictionaries containing bbox, confidence, and landmarks
        """
        # Convert BGR to RGB for PyTorch MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN with GPU acceleration
        with torch.no_grad():  # Disable gradient calculation for inference
            boxes, probs, landmarks = self.face_detector.detect(rgb_image, landmarks=True)

        detections = []
        if boxes is not None and len(boxes) > 0:
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                # Skip faces with low confidence
                if prob < self.confidence_threshold:
                    continue
                
                # Convert box from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = box.astype(int)
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

                # Extract landmarks
                landmarks_dict = {
                    'left_eye': (int(landmark[0][0]), int(landmark[0][1])),
                    'right_eye': (int(landmark[1][0]), int(landmark[1][1])),
                    'nose': (int(landmark[2][0]), int(landmark[2][1])),
                    'mouth_left': (int(landmark[3][0]), int(landmark[3][1])),
                    'mouth_right': (int(landmark[4][0]), int(landmark[4][1]))
                }

                # Calculate face quality score
                quality_score = self.assess_face_quality(rgb_image, box, landmarks_dict)

                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(prob),
                    'landmarks': landmarks_dict,
                    'quality_score': quality_score
                })

        return detections

    def assess_face_quality(self, image: np.ndarray, bbox: np.ndarray, landmarks: Dict) -> float:
        """
        Assess the quality of a detected face using multiple metrics
        
        Args:
            image: Input RGB image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            landmarks: Dictionary containing facial landmarks
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Extract face region
            x1, y1, x2, y2 = bbox.astype(int)
            face = image[y1:y2, x1:x2]
            
            if face.size == 0:
                return 0.0
                
            # 1. Size assessment - larger faces typically have better quality
            h, w = face.shape[:2]
            img_h, img_w = image.shape[:2]
            size_score = min(1.0, (w * h) / (img_w * img_h * 0.05))  # Normalize by 5% of image area
            
            # 2. Blur detection using variance of Laplacian
            gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY) if len(face.shape) == 3 else face
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            # Normalize blur score (higher variance = less blurry)
            blur_score = min(1.0, blur_score / 500.0)  # Empirical threshold
            
            # 3. Face orientation assessment using eye positions
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            
            # Calculate eye angle
            eye_angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            # Penalize if face is too tilted (ideal is 0 degrees)
            orientation_score = 1.0 - min(1.0, abs(eye_angle) / 30.0)  # Penalize angles > 30 degrees
            
            # 4. Brightness assessment
            brightness = np.mean(gray_face) / 255.0
            # Penalize too dark or too bright faces
            brightness_score = 1.0 - 2.0 * abs(brightness - 0.5)
            brightness_score = max(0.0, brightness_score)
            
            # 5. Face symmetry assessment
            # Flip the face horizontally
            flipped_face = cv2.flip(face, 1)
            # Convert to grayscale if needed
            if len(face.shape) == 3:
                gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                flipped_gray = cv2.cvtColor(flipped_face, cv2.COLOR_RGB2GRAY)
            else:
                flipped_gray = cv2.flip(gray_face, 1)
            
            # Calculate structural similarity index
            try:
                from skimage.metrics import structural_similarity as ssim
                symmetry_score, _ = ssim(gray_face, flipped_gray, full=True)
            except ImportError:
                # Fallback if skimage is not available
                # Use simple MSE-based similarity
                mse = np.mean((gray_face.astype("float") - flipped_gray.astype("float")) ** 2)
                symmetry_score = 1.0 - min(1.0, mse / 10000.0)
            
            # 6. Face coverage - check if face is cut off at image boundaries
            margin = 5  # pixels
            boundary_score = 1.0
            if (x1 <= margin or y1 <= margin or 
                x2 >= img_w - margin or y2 >= img_h - margin):
                boundary_score = 0.7  # Penalize faces at image boundaries
            
            # Weighted combination of all scores
            weights = {
                'size': 0.15,
                'blur': 0.25,
                'orientation': 0.2,
                'brightness': 0.15,
                'symmetry': 0.15,
                'boundary': 0.1
            }
            
            quality_score = (
                weights['size'] * size_score +
                weights['blur'] * blur_score +
                weights['orientation'] * orientation_score +
                weights['brightness'] * brightness_score +
                weights['symmetry'] * symmetry_score +
                weights['boundary'] * boundary_score
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            print(f"Error in face quality assessment: {e}")
            return 0.5  # Default to medium quality on error

    def align_face(self, image: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Enhanced face alignment based on eye positions using affine transformation

        Args:
            image: Input image
            landmarks: Dictionary containing facial landmarks

        Returns:
            Aligned face image
        """
        try:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            nose = np.array(landmarks['nose'])
            mouth_left = np.array(landmarks['mouth_left'])
            mouth_right = np.array(landmarks['mouth_right'])

            # Calculate angle between eyes
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))

            # Calculate desired eye distance (30% of target width)
            eye_distance = np.sqrt((dX ** 2) + (dY ** 2))
            desired_eye_distance = self.target_size[0] * 0.33  # Increased for better alignment
            scale = desired_eye_distance / eye_distance

            # Calculate eye center
            eyes_center = (int((left_eye[0] + right_eye[0]) // 2),
                           int((left_eye[1] + right_eye[1]) // 2))

            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

            # Update translation component of the matrix
            tX = self.target_size[0] * 0.5
            tY = self.target_size[1] * 0.38  # Adjusted to place eyes at optimal position
            M[0, 2] += (tX - eyes_center[0])
            M[1, 2] += (tY - eyes_center[1])

            # Apply affine transformation with improved interpolation
            aligned_face = cv2.warpAffine(
                image, M, self.target_size,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Enhanced alignment using facial landmarks for perspective correction
            # Calculate ideal positions for landmarks in the aligned face
            ideal_left_eye_pos = (int(self.target_size[0] * 0.35), int(self.target_size[1] * 0.38))
            ideal_right_eye_pos = (int(self.target_size[0] * 0.65), int(self.target_size[1] * 0.38))
            ideal_nose_pos = (int(self.target_size[0] * 0.5), int(self.target_size[1] * 0.55))
            ideal_mouth_left_pos = (int(self.target_size[0] * 0.4), int(self.target_size[1] * 0.75))
            ideal_mouth_right_pos = (int(self.target_size[0] * 0.6), int(self.target_size[1] * 0.75))
            
            # Transform the original landmarks using the affine matrix
            transformed_left_eye = self._transform_point(left_eye, M)
            transformed_right_eye = self._transform_point(right_eye, M)
            transformed_nose = self._transform_point(nose, M)
            transformed_mouth_left = self._transform_point(mouth_left, M)
            transformed_mouth_right = self._transform_point(mouth_right, M)
            
            # Source points (transformed landmarks)
            src_points = np.array([
                transformed_left_eye,
                transformed_right_eye,
                transformed_nose,
                transformed_mouth_left,
                transformed_mouth_right
            ], dtype=np.float32)
            
            # Destination points (ideal positions)
            dst_points = np.array([
                ideal_left_eye_pos,
                ideal_right_eye_pos,
                ideal_nose_pos,
                ideal_mouth_left_pos,
                ideal_mouth_right_pos
            ], dtype=np.float32)
            
            # Calculate perspective transformation matrix
            # Use a weighted approach to avoid excessive distortion
            # First, try to find a perspective transform
            try:
                # Calculate homography for perspective transform
                H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
                
                # Apply perspective transform with reduced influence (blend with original)
                perspective_face = cv2.warpPerspective(
                    aligned_face, H, self.target_size,
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )
                
                # Blend the affine and perspective transforms (70% affine, 30% perspective)
                # This prevents excessive distortion while still improving alignment
                alpha = 0.7
                final_aligned_face = cv2.addWeighted(aligned_face, alpha, perspective_face, 1-alpha, 0)
                return final_aligned_face
                
            except Exception as e:
                # If perspective transform fails, return the affine transform result
                print(f"Perspective transform failed: {e}. Using affine transform only.")
                return aligned_face
                
        except Exception as e:
            print(f"Face alignment failed: {e}. Using unaligned face.")
            # If alignment fails, just resize the extracted face
            x, y, w, h = landmarks['bbox'] if 'bbox' in landmarks else [0, 0, image.shape[1], image.shape[0]]
            face = image[y:y + h, x:x + w]
            return cv2.resize(face, self.target_size)

    def _transform_point(self, point, matrix):
        """Transform a point using an affine transformation matrix"""
        x, y = point
        transformed_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        transformed_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        return (transformed_x, transformed_y)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced normalization with adaptive histogram equalization and color correction

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        min_val, max_val = self.normalize_range
        image = image.astype(np.float32)

        # Apply advanced preprocessing techniques
        if len(image.shape) == 3:  # Color image
            # Convert to uint8 for preprocessing operations
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            
            # 1. Convert to LAB color space for better color processing
            lab_image = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_image)
            
            # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l_channel = clahe.apply(l_channel)
            
            # 3. Merge channels back and convert back to BGR
            enhanced_lab_image = cv2.merge([enhanced_l_channel, a_channel, b_channel])
            enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
            
            # 4. Apply subtle color correction to improve skin tones
            # Slightly increase red channel for better skin tone representation
            b, g, r = cv2.split(enhanced_image)
            r = np.clip(r * 1.05, 0, 255).astype(np.uint8)  # Boost red channel by 5%
            enhanced_image = cv2.merge([b, g, r])
            
            # 5. Apply subtle bilateral filtering to reduce noise while preserving edges
            enhanced_image = cv2.bilateralFilter(enhanced_image, 5, 35, 35)
            
            # Convert back to float32 for normalization
            image = enhanced_image.astype(np.float32)
        else:  # Grayscale image
            # Make sure image is in uint8 format for cv2.equalizeHist
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            
            # Apply CLAHE for grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(img_uint8).astype(np.float32)

        # Normalize to target range
        if min_val == -1 and max_val == 1:
            # Normalize to [-1, 1] range for better neural network performance
            image = (image / 127.5) - 1
        else:
            # Normalize to [0, 1] range
            image = image / 255.0
            # Scale to target range if not [0, 1]
            if min_val != 0 or max_val != 1:
                image = image * (max_val - min_val) + min_val

        return image

    def extract_face(self, image: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Extract face from image using detection bbox with improved margin handling

        Args:
            image: Input image
            detection: Face detection data with bbox

        Returns:
            Extracted face image
        """
        try:
            x, y, w, h = detection['bbox']

            # Add margin based on class parameter
            margin_x = int(w * self.margin_percent)
            margin_y = int(h * self.margin_percent)

            # Ensure coordinates stay within image bounds
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)

            # Extract face region with margin
            face = image[y1:y2, x1:x2]

            # If extraction failed or resulted in empty image, return resized original image
            if face.size == 0:
                print("Warning: Face extraction resulted in empty image. Using full image.")
                return cv2.resize(image, self.target_size)

            return face
        except Exception as e:
            print(f"Error in face extraction: {e}. Using full image.")
            return cv2.resize(image, self.target_size)

    def preprocess_face(self, image: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Complete face preprocessing pipeline with enhanced alignment, background removal, and normalization

        Args:
            image: Input image
            detection: Detection data with bbox and landmarks

        Returns:
            Preprocessed face image
        """
        # Check face quality and skip low-quality faces if below threshold
        if 'quality_score' in detection and detection['quality_score'] < self.quality_threshold:
            print(f"Low quality face detected (score: {detection['quality_score']:.2f}). Using enhanced processing.")
            # For low quality faces, we'll still process them but with additional enhancements
        
        # Extract face with margin
        face = self.extract_face(image, detection)
        
        # Align face using landmarks if possible
        try:
            aligned_face = self.align_face(image, detection['landmarks'])
        except Exception as e:
            print(f"Face alignment failed: {e}. Using unaligned face.")
            # If alignment fails, just resize the extracted face
            aligned_face = cv2.resize(face, self.target_size)
    
        # Remove background to focus on facial features
        face_no_bg = self.remove_background(aligned_face)
        
        # Normalize pixel values with enhanced preprocessing
        normalized_face = self.normalize_image(face_no_bg)
        
        return normalized_face

    def process_image(self, 
                      image: np.ndarray, 
                      return_all_faces: bool = False) -> Union[np.ndarray, List[np.ndarray], None]:
        """
        Process an image and return preprocessed face(s)

        Args:
            image: Input image
            return_all_faces: If True, return all detected faces

        Returns:
            Single preprocessed face or list of preprocessed faces
        """
        # Detect faces
        detections = self.detect_faces(image)
        if not detections:
            return None

        # Filter out low-quality faces if we have multiple faces
        if len(detections) > 1:
            filtered_detections = [d for d in detections if d['quality_score'] >= self.quality_threshold]
            # If all faces are filtered out, keep the highest quality one
            if not filtered_detections:
                highest_quality_idx = max(range(len(detections)), key=lambda i: detections[i]['quality_score'])
                filtered_detections = [detections[highest_quality_idx]]
            detections = filtered_detections

        # Process each detected face
        processed_faces = []
        for detection in detections:
            processed_face = self.preprocess_face(image, detection)
            processed_faces.append(processed_face)

        if return_all_faces:
            return processed_faces
        
        # Return the face with highest confidence if not returning all
        if len(detections) > 1:
            # If we have quality scores, use them as the primary criterion
            highest_quality_idx = max(range(len(detections)), 
                                     key=lambda i: detections[i]['quality_score'])
            return processed_faces[highest_quality_idx]
        else:
            # If only one face, return it
            return processed_faces[0]

    def process_directory(self,
                      input_dir: str,
                      save_dir: str = None) -> List[ProcessedFace]:
        """
        Process all images in a directory with parallel processing

        Args:
            input_dir: Directory containing input images
            save_dir: Optional directory to save processed faces

        Returns:
            List of ProcessedFace objects containing processed faces and their metadata
        """
        processed_faces = []
        failed_images = []

        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Process each image in the directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                image_path = os.path.join(input_dir, filename)
                try:
                    # Try reading with OpenCV first
                    image = cv2.imread(image_path)
                    
                    # If OpenCV fails, try alternative methods
                    if image is None:
                        print(f"OpenCV failed to read image: {filename}, trying alternative methods...")
                        
                        # Check if file exists and has content
                        if not os.path.exists(image_path):
                            print(f"Error: File does not exist: {filename}")
                            failed_images.append((filename, "File does not exist"))
                            continue
                            
                        file_size = os.path.getsize(image_path)
                        if file_size == 0:
                            print(f"Error: File is empty (0 bytes): {filename}")
                            failed_images.append((filename, "Empty file (0 bytes)"))
                            continue
                        
                        # Try PIL/Pillow
                        try:
                            from PIL import Image, UnidentifiedImageError
                            try:
                                pil_image = Image.open(image_path)
                                img_format = pil_image.format
                                pil_image = pil_image.convert('RGB')
                                image = np.array(pil_image)
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                print(f"Successfully loaded image using PIL: {filename} (Format: {img_format})")
                            except UnidentifiedImageError:
                                print(f"PIL Error: Unidentified image format in file: {filename}")
                                failed_images.append((filename, "Unidentified image format"))
                                continue
                            except Exception as pil_error:
                                print(f"PIL Error: {str(pil_error)} when reading: {filename}")
                                
                                # Try imageio as a last resort
                                try:
                                    import imageio
                                    image = imageio.imread(image_path)
                                    if len(image.shape) == 3 and image.shape[2] == 3:
                                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                    print(f"Successfully loaded image using imageio: {filename}")
                                except ImportError:
                                    print("imageio library not available for fallback image loading")
                                    failed_images.append((filename, f"PIL error: {str(pil_error)}, imageio not available"))
                                    continue
                                except Exception as imageio_error:
                                    print(f"All image loading methods failed for: {filename}")
                                    print(f"Detailed error: {str(imageio_error)}")
                                    failed_images.append((filename, f"All loading methods failed: {str(imageio_error)}"))
                                    continue
                        except ImportError:
                            print("PIL/Pillow library not available for fallback image loading")
                            failed_images.append((filename, "OpenCV failed and PIL not available"))
                            continue
                    
                    if image is None:
                        print(f"All image loading methods failed for: {filename}")
                        failed_images.append((filename, "Unknown image loading failure"))
                        continue

                    # Detect and process faces
                    detections = self.detect_faces(image)

                    if not detections:
                        print(f"No faces detected in image: {filename}")
                        failed_images.append((filename, "No faces detected"))
                        continue

                    for idx, detection in enumerate(detections):
                        # Skip low-quality faces unless it's the only face
                        if (detection['quality_score'] < self.quality_threshold and 
                            len(detections) > 1):
                            print(f"Skipping low quality face in {filename} (score: {detection['quality_score']:.2f})")
                            continue
                            
                        # Process face
                        processed_face_img = self.preprocess_face(image, detection)

                        # Create ProcessedFace object
                        processed_face = ProcessedFace(
                            face_image=processed_face_img,
                            original_filename=filename,
                            confidence=detection['confidence'],
                            face_id=idx,
                            bbox=tuple(detection['bbox']),
                            quality_score=detection['quality_score']
                        )

                        processed_faces.append(processed_face)

                        # Save processed face if directory is specified
                        if save_dir:
                            base_name = Path(filename).stem
                            # Save as numpy array
                            np_save_path = os.path.join(
                                save_dir,
                                f"{base_name}_face_{idx}.npy"
                            )
                            np.save(np_save_path, processed_face_img)
                            
                            # Also save as image for visualization
                            img_save_path = os.path.join(
                                save_dir,
                                f"{base_name}_face_{idx}.jpg"
                            )
                            # Convert from normalized to 0-255 range for saving
                            save_img = ((processed_face_img - self.normalize_range[0]) / 
                                        (self.normalize_range[1] - self.normalize_range[0]) * 255).astype(np.uint8)
                            cv2.imwrite(img_save_path, save_img)

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    failed_images.append((filename, f"Processing error: {str(e)}"))

        # Log summary of failed images
        if failed_images:
            print(f"\nFailed to process {len(failed_images)} images:")
            for filename, reason in failed_images:
                print(f"  - {filename}: {reason}")
        
        return processed_faces

    def preprocess_image(self, image_path):
        """
        Preprocess a single image with timeout protection and enhanced error handling
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed face image or None if no face detected
        """
        try:
            # Try reading with OpenCV first
            image = cv2.imread(image_path)
            
            # If OpenCV fails, try alternative methods
            if image is None:
                print(f"OpenCV failed to read image: {image_path}, trying alternative methods...")
                
                # Check if file exists
                if not os.path.exists(image_path):
                    print(f"Error: File does not exist: {image_path}")
                    return None
                    
                # Check file size
                file_size = os.path.getsize(image_path)
                if file_size == 0:
                    print(f"Error: File is empty (0 bytes): {image_path}")
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
                        print(f"Successfully loaded image using PIL: {image_path} (Format: {img_format})")
                    except UnidentifiedImageError:
                        print(f"PIL Error: Unidentified image format in file: {image_path}")
                        return None
                    except Exception as pil_error:
                        print(f"PIL Error: {str(pil_error)} when reading: {image_path}")
                        
                        # Try imageio as a last resort
                        try:
                            import imageio
                            image = imageio.imread(image_path)
                            # Convert to BGR for OpenCV if needed
                            if len(image.shape) == 3 and image.shape[2] == 3:
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            print(f"Successfully loaded image using imageio: {image_path}")
                        except ImportError:
                            print("imageio library not available for fallback image loading")
                            return None
                        except Exception as imageio_error:
                            print(f"All image loading methods failed for: {image_path}")
                            print(f"Detailed error: {str(imageio_error)}")
                            
                            # Provide diagnostic information about the file
                            try:
                                import magic
                                file_type = magic.from_file(image_path)
                                print(f"File type according to magic: {file_type}")
                            except ImportError:
                                print("python-magic library not available for file type detection")
                                
                            print(f"File size: {file_size} bytes")
                            return None
                except ImportError:
                    print("PIL/Pillow library not available for fallback image loading")
                    return None
            
            if image is None:
                print(f"All image loading methods failed for: {image_path}")
                return None
            
            # Process image using the enhanced pipeline
            return self.process_image(image)
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_preprocessing(self, image: np.ndarray, save_dir: str = "preprocessing_steps") -> None:
        """
        Visualize and save each step of the preprocessing pipeline

        Args:
            image: Input image in BGR format
            save_dir: Directory to save the visualization steps
        """
        os.makedirs(save_dir, exist_ok=True)

        try:
            # Step 1: Save original image
            cv2.imwrite(os.path.join(save_dir, "1_original.jpg"), image)

            # Step 2: Convert to RGB (for MTCNN) and save
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, "2_rgb_converted.jpg"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

            # Step 3: Detect faces and draw bounding boxes
            detections = self.detect_faces(image)
            visualization = image.copy()

            if not detections:
                print("No faces detected for visualization")
                cv2.putText(visualization, "No faces detected", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(save_dir, "3_no_face_detected.jpg"), visualization)
                return

            for det in detections:
                x, y, w, h = det['bbox']

                # Draw face bounding box
                cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw confidence score
                conf_text = f"Conf: {det['confidence']:.2f}"
                cv2.putText(visualization, conf_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw quality score
                quality_text = f"Quality: {det['quality_score']:.2f}"
                cv2.putText(visualization, quality_text, (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 255, 0) if det['quality_score'] >= self.quality_threshold else (0, 0, 255), 2)

                # Draw landmarks
                for point_name, point in det['landmarks'].items():
                    color_map = {
                        'left_eye': (255, 0, 0),  # Blue
                        'right_eye': (255, 0, 0),  # Blue
                        'nose': (0, 255, 0),  # Green
                        'mouth_left': (0, 0, 255),  # Red
                        'mouth_right': (0, 0, 255)  # Red
                    }
                    cv2.circle(visualization, point, 3, color_map[point_name], -1)

            cv2.imwrite(os.path.join(save_dir, "3_face_detection.jpg"), visualization)

            # Process each detected face
            for idx, det in enumerate(detections):
                try:
                    # Step 4a: Extract face with margin
                    face_img = self.extract_face(image, det)
                    cv2.imwrite(os.path.join(save_dir, f"4a_extracted_face_{idx}.jpg"), face_img)

                    # Step 4b: Save aligned face
                    try:
                        aligned_face = self.align_face(image, det['landmarks'])
                        cv2.imwrite(os.path.join(save_dir, f"4b_aligned_face_{idx}.jpg"), aligned_face)
                    except Exception as e:
                        print(f"Warning: Face alignment failed during visualization: {e}")
                        # Save unaligned face instead
                        cv2.imwrite(os.path.join(save_dir, f"4b_unaligned_face_{idx}.jpg"), face_img)

                    # New Step: Save face with background removed
                    try:
                        aligned_face = self.align_face(image, det['landmarks'])
                        face_no_bg = self.remove_background(aligned_face)
                        # Convert to visualization format
                        if self.normalize_range[0] < 0:
                            face_no_bg_vis = ((face_no_bg - self.normalize_range[0]) / 
                                             (self.normalize_range[1] - self.normalize_range[0]) * 255).astype(np.uint8)
                        else:
                            face_no_bg_vis = (face_no_bg * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(save_dir, f"5b_background_removed_{idx}.jpg"), face_no_bg_vis)
                    except Exception as e:
                        print(f"Warning: Background removal failed during visualization: {e}")

                    # Step 5: Save resized face (if not already done by alignment)
                    resized_face = cv2.resize(face_img, self.target_size)
                    cv2.imwrite(os.path.join(save_dir, f"5_resized_face_{idx}.jpg"), resized_face)

                    # Step 6: Save normalized face (without histogram equalization for visualization)
                    # Simple normalization for visualization
                    normalized_vis = cv2.normalize(resized_face, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, f"6_normalized_face_{idx}.jpg"), normalized_vis)

                    # Step 7: Full normalization (with proper error handling)
                    try:
                        normalized_face = self.normalize_image(resized_face)
                        # Convert back to 0-255 range for visualization
                        full_normalized_vis = ((normalized_face - self.normalize_range[0]) /
                                               (self.normalize_range[1] - self.normalize_range[0]) * 255).astype(
                            np.uint8)
                        cv2.imwrite(os.path.join(save_dir, f"7_full_normalized_face_{idx}.jpg"), full_normalized_vis)
                    except Exception as e:
                        print(f"Warning: Full normalization failed: {e}")

                except Exception as e:
                    print(f"Error processing face {idx}: {e}")

            print(f"Preprocessing visualization saved to {save_dir}")

        except Exception as e:
            print(f"Error during visualization: {e}")

    def remove_background(self, face_image: np.ndarray) -> np.ndarray:
        """
        Advanced background removal using semantic segmentation and blending
        
        Args:
            face_image: Aligned face image
            
        Returns:
            Face image with background removed/blurred
        """
        try:
            # Convert image to appropriate format if needed
            if face_image.dtype != np.uint8:
                # If normalized to [-1,1], convert back to [0,255]
                if self.normalize_range[0] < 0:
                    temp_img = ((face_image - self.normalize_range[0]) / 
                            (self.normalize_range[1] - self.normalize_range[0]) * 255).astype(np.uint8)
                else:
                    temp_img = (face_image * 255).astype(np.uint8)
            else:
                temp_img = face_image.copy()
                
            # Get image dimensions
            height, width = temp_img.shape[:2]
            
            # Step 1: Create a base mask using facial landmarks-based segmentation
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create elliptical mask centered on the face
            center_x, center_y = width // 2, height // 2
            # Ellipse axes (face is typically taller than wide)
            axes_length = (int(width * 0.42), int(height * 0.55))
            # Draw filled white ellipse on black background
            cv2.ellipse(mask, (center_x, center_y), axes_length, 
                    0, 0, 360, (255), -1)
            
            # Step 2: Refine the mask using color-based segmentation for skin detection
            # Convert to YCrCb color space which is better for skin detection
            if len(temp_img.shape) == 3:  # Color image
                ycrcb_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2YCrCb)
                # Define skin color range in YCrCb
                lower_skin = np.array([0, 135, 85], dtype=np.uint8)
                upper_skin = np.array([255, 180, 135], dtype=np.uint8)
                # Create skin mask
                skin_mask = cv2.inRange(ycrcb_img, lower_skin, upper_skin)
                
                # Combine the elliptical mask with skin detection mask
                combined_mask = cv2.bitwise_and(mask, skin_mask)
                
                # Apply morphological operations to clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                
                # Dilate to include more of the face
                combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
            else:
                # For grayscale images, just use the elliptical mask
                combined_mask = mask
            
            # Step 3: Apply GrabCut algorithm for more precise segmentation
            try:
                if len(temp_img.shape) == 3:  # GrabCut only works on color images
                    # Create GrabCut mask
                    grabcut_mask = np.zeros(temp_img.shape[:2], dtype=np.uint8)
                    # Set combined_mask area as probable foreground
                    grabcut_mask[combined_mask > 0] = cv2.GC_PR_FGD
                    # Set outer area as probable background
                    border = 10
                    grabcut_mask[:border, :] = cv2.GC_BGD
                    grabcut_mask[-border:, :] = cv2.GC_BGD
                    grabcut_mask[:, :border] = cv2.GC_BGD
                    grabcut_mask[:, -border:] = cv2.GC_BGD
                    
                    # Apply GrabCut
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)
                    rect = (border, border, width-2*border, height-2*border)
                    cv2.grabCut(temp_img, grabcut_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
                    
                    # Create final mask
                    final_mask = np.where((grabcut_mask == cv2.GC_PR_FGD) | (grabcut_mask == cv2.GC_FGD), 255, 0).astype('uint8')
                else:
                    final_mask = combined_mask
            except Exception as e:
                print(f"GrabCut segmentation failed: {e}. Using simpler mask.")
                final_mask = combined_mask
            
            # Step 4: Apply Gaussian blur to the mask edges for smoother transition
            final_mask = cv2.GaussianBlur(final_mask, (15, 15), 0)
            
            # Normalize mask to range [0, 1]
            mask_norm = final_mask.astype(float) / 255.0
            
            # Expand mask dimensions for broadcasting if image is color
            if len(temp_img.shape) == 3:
                mask_norm = np.expand_dims(mask_norm, axis=2)
                
            # Step 5: Create background blur for smoother transition
            blurred_img = cv2.GaussianBlur(temp_img, (25, 25), 0)
            
            # Step 6: Blend original image and blurred image using the mask
            # Keep face region, blur background
            if len(temp_img.shape) == 3:
                result = (temp_img * mask_norm + blurred_img * (1 - mask_norm)).astype(np.uint8)
            else:
                result = (temp_img * mask_norm + blurred_img * (1 - mask_norm)).astype(np.uint8)
                
            # Convert back to original format if needed
            if face_image.dtype != np.uint8:
                if self.normalize_range[0] < 0:
                    # Convert back to [-1,1] range
                    result = (result / 127.5) - 1
                else:
                    # Convert back to [0,1] range
                    result = result / 255.0
                    
            return result
            
        except Exception as e:
            print(f"Advanced background removal failed: {e}. Using original face image.")
            return face_image