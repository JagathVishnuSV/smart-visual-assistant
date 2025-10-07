"""
Object Detection Module - YOLOv8 Nano for low-compute systems
"""
from ultralytics import YOLO
import cv2
import logging
from typing import Tuple, List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """Lightweight object detector using YOLOv8 Nano"""
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model (nano is smallest/fastest)
            conf_threshold: Confidence threshold for detections
        """
        try:
            logger.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            logger.info("Object detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise
    
    def detect(self, frame: np.ndarray, img_size=640) -> Tuple[np.ndarray, List[str]]:
        """
        Detect objects in frame
        
        Args:
            frame: Input image (numpy array)
            img_size: Input size for model (smaller = faster)
        
        Returns:
            Tuple of (annotated frame, list of unique object labels)
        """
        try:
            # Run inference with optimizations for low-compute
            results = self.model(
                frame,
                imgsz=img_size,
                conf=self.conf_threshold,
                verbose=False,
                device='cpu'  # Force CPU for compatibility
            )[0]
            
            # Get annotated frame
            annotated = results.plot()
            
            # Extract unique object labels
            objects = []
            if len(results.boxes) > 0:
                class_ids = results.boxes.cls.cpu().numpy()
                objects = [self.model.names[int(cls)] for cls in class_ids]
                objects = list(set(objects))  # Remove duplicates
            
            return annotated, objects
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return frame, []
    
    def detect_with_counts(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Detect objects and return with counts
        
        Returns:
            Tuple of (annotated frame, dict of {object: count})
        """
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            annotated = results.plot()
            
            object_counts = {}
            if len(results.boxes) > 0:
                class_ids = results.boxes.cls.cpu().numpy()
                for cls_id in class_ids:
                    label = self.model.names[int(cls_id)]
                    object_counts[label] = object_counts.get(label, 0) + 1
            
            return annotated, object_counts
            
        except Exception as e:
            logger.error(f"Detection with counts failed: {e}")
            return frame, {}