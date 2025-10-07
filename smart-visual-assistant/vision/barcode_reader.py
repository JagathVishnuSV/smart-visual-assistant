"""
Barcode and QR Code Detection Module
Uses pyzbar for efficient barcode/QR code reading
"""
import cv2
import logging
from typing import Tuple, List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_barcodes(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Detect and decode barcodes and QR codes
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        Tuple of (annotated image, list of decoded codes)
    """
    try:
        from pyzbar.pyzbar import decode
        
        barcodes = decode(image)
        codes = []
        
        for barcode in barcodes:
            # Extract barcode data
            x, y, w, h = barcode.rect
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            
            codes.append(f"{barcode_type}: {barcode_data}")
            
            # Draw rectangle around barcode
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            # Add text label
            label = f"{barcode_type}: {barcode_data[:20]}"  # Truncate long codes
            cv2.putText(
                image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
            )
        
        return image, codes
        
    except ImportError:
        logger.error("pyzbar not installed. Install with: pip install pyzbar")
        return image, []
    except Exception as e:
        logger.error(f"Barcode detection failed: {e}")
        return image, []


def detect_qr_codes(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Detect QR codes specifically using OpenCV's QRCodeDetector
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        Tuple of (annotated image, list of decoded QR codes)
    """
    try:
        qr_detector = cv2.QRCodeDetector()
        data, bbox, _ = qr_detector.detectAndDecode(image)
        
        codes = []
        
        if data:
            codes.append(data)
            
            # Draw bounding box if available
            if bbox is not None:
                bbox = bbox.astype(int)
                for i in range(len(bbox[0])):
                    pt1 = tuple(bbox[0][i])
                    pt2 = tuple(bbox[0][(i + 1) % len(bbox[0])])
                    cv2.line(image, pt1, pt2, (0, 255, 0), 3)
                
                # Add text
                cv2.putText(
                    image, data[:30], tuple(bbox[0][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
        
        return image, codes
        
    except Exception as e:
        logger.error(f"QR code detection failed: {e}")
        return image, []
