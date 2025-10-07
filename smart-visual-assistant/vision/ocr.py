"""
OCR Module - Text extraction from images
Uses EasyOCR with GPU disabled for low-compute systems
"""
import cv2
import logging
from typing import Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reader instance (lazy loaded)
_reader = None


def get_reader():
    """Lazy load EasyOCR reader"""
    global _reader
    if _reader is None:
        try:
            import easyocr
            logger.info("Loading EasyOCR (this may take a moment)...")
            _reader = easyocr.Reader(['en'], gpu=False)  # CPU mode for compatibility
            logger.info("EasyOCR loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            raise
    return _reader


def extract_text(image: np.ndarray, min_confidence=0.5) -> Tuple[np.ndarray, str]:
    """
    Extract text from image using OCR
    
    Args:
        image: Input image (BGR format)
        min_confidence: Minimum confidence threshold for text detection
    
    Returns:
        Tuple of (annotated image, extracted text)
    """
    try:
        reader = get_reader()
        results = reader.readtext(image)
        full_text = []
        
        for (bbox, text, prob) in results:
            # Only include high-confidence detections
            if prob >= min_confidence:
                # Draw bounding box
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(
                    image, text, top_left,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                
                full_text.append(text)
        
        result_text = " ".join(full_text) if full_text else ""
        return image, result_text
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return image, ""


def extract_text_simple(image: np.ndarray) -> str:
    """
    Extract text only (no image annotation)
    
    Args:
        image: Input image
    
    Returns:
        Extracted text string
    """
    try:
        reader = get_reader()
        results = reader.readtext(image)
        texts = [text for (_, text, prob) in results if prob >= 0.5]
        return " ".join(texts)
    except Exception as e:
        logger.error(f"Simple OCR failed: {e}")
        return ""
