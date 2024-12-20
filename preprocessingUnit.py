import cv2
import numpy as np
import dlib
from mtcnn import MTCNN
import os
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessedFace:
    """Data class to store processed face information"""
    face_image: np.ndarray
    original_filename: str
    confidence: float
    face_id: int  # To distinguish multiple faces from same image
    bbox: Tuple[int, int, int, int]


class FacePreprocessor:
    # def __init__(self,
    #              target_size: Tuple[int, int] = (224, 224),
    #              use_mtcnn: bool = True,
    #              normalize_range: Tuple[float, float] = (0, 1)):

    def __init__(self, target_size=(224, 224), use_mtcnn=True, normalize_range=(0, 1)):
        self.target_size = target_size
        self.normalize_range = normalize_range
        self.face_detector = MTCNN() if use_mtcnn else None
        """
        Initialize the face preprocessing pipeline

        Args:
            target_size: Output size for face images
            use_mtcnn: If True, use MTCNN for face detection/alignment
            normalize_range: Range for pixel normalization
        """
        self.target_size = target_size
        self.normalize_range = normalize_range

        # Initialize face detector
        if use_mtcnn:
            self.face_detector = MTCNN()
        else:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(
                'shape_predictor_68_face_landmarks.dat'
            )

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image and return their bounding boxes and landmarks
        """
        if isinstance(self.face_detector, MTCNN):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.face_detector.detect_faces(rgb_image)

            return [{
                'bbox': d['box'],
                'confidence': d['confidence'],
                'landmarks': {
                    'left_eye': d['keypoints']['left_eye'],
                    'right_eye': d['keypoints']['right_eye'],
                    'nose': d['keypoints']['nose'],
                    'mouth_left': d['keypoints']['mouth_left'],
                    'mouth_right': d['keypoints']['mouth_right']
                }
            } for d in detections]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dlib_rects = self.face_detector(gray)

            detections = []
            for rect in dlib_rects:
                shape = self.landmark_predictor(gray, rect)
                landmarks = self._shape_to_dict(shape)

                detections.append({
                    'bbox': [rect.left(), rect.top(),
                             rect.width(), rect.height()],
                    'confidence': 1.0,
                    'landmarks': landmarks
                })

            return detections

    def align_face(self,
                   image: np.ndarray,
                   landmarks: Dict) -> np.ndarray:
        """
        Align face based on eye positions
        """
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        eye_distance = np.sqrt((dX ** 2) + (dY ** 2))
        desired_eye_distance = self.target_size[0] * 0.3
        scale = desired_eye_distance / eye_distance

        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tX = self.target_size[0] * 0.5
        tY = self.target_size[1] * 0.35
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        aligned_face = cv2.warpAffine(image, M, self.target_size,
                                      flags=cv2.INTER_CUBIC)

        return aligned_face

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to specified range
        """
        min_val, max_val = self.normalize_range
        image = image.astype(np.float32)

        if min_val == -1 and max_val == 1:
            image = (image / 127.5) - 1
        else:
            image = image / 255.0

        return image

    def process_directory(self,
                          input_dir: str,
                          save_dir: str = None) -> List[ProcessedFace]:
        """
        Process all images in a directory and maintain filename mapping

        Args:
            input_dir: Directory containing input images
            save_dir: Optional directory to save processed faces

        Returns:
            List of ProcessedFace objects containing processed faces and their metadata
        """
        processed_faces = []

        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Process each image in the directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Could not load image: {filename}")
                        continue

                    # Detect and process faces
                    detections = self.detect_faces(image)

                    for idx, detection in enumerate(detections):
                        # Align and normalize face
                        aligned_face = self.align_face(image, detection['landmarks'])
                        normalized_face = self.normalize_image(aligned_face)

                        # Create ProcessedFace object
                        processed_face = ProcessedFace(
                            face_image=normalized_face,
                            original_filename=filename,
                            confidence=detection['confidence'],
                            face_id=idx,
                            bbox=tuple(detection['bbox'])
                        )

                        processed_faces.append(processed_face)

                        # Save processed face if directory is specified
                        if save_dir:
                            base_name = Path(filename).stem
                            save_path = os.path.join(
                                save_dir,
                                f"{base_name}_face_{idx}.npy"
                            )
                            np.save(save_path, normalized_face)

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        return processed_faces

    def save_mapping(self,
                     processed_faces: List[ProcessedFace],
                     output_file: str):
        """
        Save mapping between processed faces and original filenames
        """
        mapping = []
        for face in processed_faces:
            mapping.append({
                'original_filename': face.original_filename,
                'face_id': face.face_id,
                'confidence': face.confidence,
                'bbox': face.bbox
            })

        np.save(output_file, mapping)

    def preprocess_image(self, image_path: str, return_all_faces: bool = False) -> np.ndarray:
        """
        Preprocess an image for face recognition
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Detect faces
        detections = self.detect_faces(image)
        if not detections:
            raise ValueError(f"No faces detected in image: {image_path}")

        processed_faces = []
        for face_data in detections:
            x, y, w, h = face_data['bbox']

            # Extract face region
            face = image[y:y + h, x:x + w]

            # Resize to target size
            face = cv2.resize(face, self.target_size)

            # Normalize pixel values
            min_val, max_val = self.normalize_range
            face = (face - 127.5) / 127.5 * (max_val - min_val) + min_val

            processed_faces.append(face)

        if return_all_faces:
            return np.array(processed_faces)
        return np.array(processed_faces[0])

    def visualize_preprocessing(self, image: np.ndarray, save_dir: str = "preprocessing_steps") -> None:
        """
        Visualize and save each step of the preprocessing pipeline
        
        Args:
            image: Input image in BGR format
            save_dir: Directory to save the visualization steps
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Step 1: Save original image
        cv2.imwrite(os.path.join(save_dir, "1_original.jpg"), image)
        
        # Step 2: Convert to RGB and save
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_dir, "2_rgb_converted.jpg"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        # Step 3: Detect faces and draw bounding boxes
        detections = self.detect_faces(image)
        visualization = image.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw landmarks
            for point_name, point in det['landmarks'].items():
                cv2.circle(visualization, point, 3, (255, 0, 0), -1)
                
        cv2.imwrite(os.path.join(save_dir, "3_face_detection.jpg"), visualization)
        
        # Step 4: Save each detected face
        for idx, det in enumerate(detections):
            face_img = image[det['bbox'][1]:det['bbox'][1]+det['bbox'][3], 
                            det['bbox'][0]:det['bbox'][0]+det['bbox'][2]]
            cv2.imwrite(os.path.join(save_dir, f"4_extracted_face_{idx}.jpg"), face_img)
            
            # Step 5: Save resized face
            resized_face = cv2.resize(face_img, self.target_size)
            cv2.imwrite(os.path.join(save_dir, f"5_resized_face_{idx}.jpg"), resized_face)
            
            # Step 6: Save normalized face
            normalized_face = (resized_face - self.normalize_range[0]) / (self.normalize_range[1] - self.normalize_range[0])
            normalized_vis = (normalized_face * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"6_normalized_face_{idx}.jpg"), normalized_vis)
