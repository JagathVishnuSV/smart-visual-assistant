"""
Face Detection Module - Using OpenCV Haar Cascades (lightweight)
Switched from face_recognition library to OpenCV for low-compute systems
"""
import cv2
import logging
from typing import Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Lightweight face detector using OpenCV Haar Cascades"""
    
    def __init__(self):
        """Initialize face detector"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray, detect_eyes=False) -> Tuple[np.ndarray, int]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
            detect_eyes: Also detect eyes (more processing)
        
        Returns:
            Tuple of (annotated image, face count)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if detect_eyes:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = image[y:y+h, x:x+w]
                    
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
            
            return image, len(faces)
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return image, 0


# Standalone function for backward compatibility
def detect_faces(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Legacy function - detects faces using OpenCV Haar Cascades
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        Tuple of (annotated image, face count)
    """
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in face_locations:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return image, len(face_locations)
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return image, 0