"""
Unified Vision Model Manager - Optimized for Low-Compute Systems
Lazy-loads models only when needed to minimize memory usage
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastVLM:
    """Unified vision processing with lazy model loading for memory efficiency"""
    
    def __init__(self):
        # Models are loaded lazily to save memory
        self._detector_model = None
        self._face_detector = None
        self._ocr_reader = None
        self._clip_model = None
        self._clip_preprocess = None
        
    @property
    def detector(self):
        """Lazy load YOLO detector"""
        if self._detector_model is None:
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLO model...")
                self._detector_model = YOLO("yolov8n.pt")  # Nano model for low compute
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                raise
        return self._detector_model
    
    @property
    def ocr_reader(self):
        """Lazy load OCR reader"""
        if self._ocr_reader is None:
            try:
                import easyocr
                logger.info("Loading EasyOCR...")
                self._ocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU mode
                logger.info("EasyOCR loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                raise
        return self._ocr_reader
    
    @property
    def clip_model(self):
        """Lazy load CLIP model"""
        if self._clip_model is None:
            try:
                import torch
                import clip
                logger.info("Loading CLIP model...")
                device = "cpu"  # Force CPU for low compute
                self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=device)
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
                raise
        return self._clip_model, self._clip_preprocess
    
    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Detect objects in image using YOLO"""
        try:
            # Resize for faster processing on low-compute systems
            height, width = image.shape[:2]
            if width > 640:
                scale = 640 / width
                image = cv2.resize(image, (640, int(height * scale)))
            
            results = self.detector(image, verbose=False)[0]
            annotated = results.plot()
            
            # Get unique object labels
            objects = []
            if len(results.boxes) > 0:
                objects = [results.names[int(cls)] for cls in results.boxes.cls]
                objects = list(set(objects))
            
            return annotated, objects
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return image, []
    
    def detect_faces(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Detect faces using OpenCV Haar Cascades (lightweight)"""
        try:
            # Use Haar cascades instead of face_recognition for low compute
            if self._face_detector is None:
                self._face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            return image, len(faces)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return image, 0
    
    def detect_barcodes(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Detect and decode barcodes/QR codes"""
        try:
            from pyzbar.pyzbar import decode
            barcodes = decode(image)
            codes = []
            
            for barcode in barcodes:
                x, y, w, h = barcode.rect
                text = barcode.data.decode("utf-8")
                codes.append(text)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(image, text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            return image, codes
        except Exception as e:
            logger.error(f"Barcode detection failed: {e}")
            return image, []
    
    def extract_text(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Extract text using OCR"""
        try:
            results = self.ocr_reader.readtext(image)
            full_text = ""
            
            for (bbox, text, prob) in results:
                if prob > 0.5:  # Only high confidence text
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(image, text, top_left, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    full_text += text + " "
            
            return image, full_text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return image, ""
    
    def detect_dominant_color(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Detect dominant color in image"""
        try:
            from sklearn.cluster import KMeans
            
            # Sample pixels for faster processing
            img_sample = cv2.resize(image, (150, 150))
            img = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
            img = img.reshape((-1, 3))
            
            kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
            kmeans.fit(img)
            
            dominant = kmeans.cluster_centers_[0].astype(int)
            color_rgb = tuple(dominant)
            
            # Draw color box
            color_bgr = color_rgb[::-1]
            cv2.rectangle(image, (10, 10), (110, 110), color_bgr, -1)
            cv2.rectangle(image, (10, 10), (110, 110), (0, 0, 0), 2)
            
            # Color name approximation
            color_name = self._get_color_name(color_rgb)
            
            return image, f"{color_name} (RGB: {color_rgb})"
        except Exception as e:
            logger.error(f"Color detection failed: {e}")
            return image, "Unknown"
    
    def answer_question(self, image_np: np.ndarray, question: str) -> Tuple[np.ndarray, str]:
        """Answer questions about image using CLIP"""
        try:
            import torch
            from PIL import Image
            
            model, preprocess = self.clip_model
            
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            image_input = preprocess(pil_image).unsqueeze(0)
            
            # Predefined candidates for common objects
            candidates = [
                "person", "people", "face", "cup", "phone", "laptop", 
                "nothing", "book", "pen", "bottle", "table", "chair",
                "keyboard", "mouse", "monitor", "bag", "door", "window",
                "car", "bicycle", "dog", "cat", "food", "drink"
            ]
            
            texts = [f"a photo of a {item}" for item in candidates]
            text_tokens = torch.clip.tokenize(texts)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(3)
            
            # Get top 3 predictions
            predictions = [candidates[idx] for idx in indices]
            answer = f"I see: {predictions[0]} (also possibly: {predictions[1]}, {predictions[2]})"
            
            return image_np, answer
        except Exception as e:
            logger.error(f"CLIP Q&A failed: {e}")
            return image_np, "Could not analyze the image"
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Approximate color name from RGB"""
        r, g, b = rgb
        
        if r > 200 and g > 200 and b > 200:
            return "White"
        elif r < 50 and g < 50 and b < 50:
            return "Black"
        elif r > 150 and g < 100 and b < 100:
            return "Red"
        elif r < 100 and g > 150 and b < 100:
            return "Green"
        elif r < 100 and g < 100 and b > 150:
            return "Blue"
        elif r > 150 and g > 150 and b < 100:
            return "Yellow"
        elif r > 150 and g < 100 and b > 150:
            return "Magenta"
        elif r < 100 and g > 150 and b > 150:
            return "Cyan"
        elif r > 150 and g > 100 and b < 100:
            return "Orange"
        elif r > 100 and g < 100 and b > 100:
            return "Purple"
        elif r > 100 and g > 100 and b > 100:
            return "Gray"
        else:
            return "Mixed"
    
    def cleanup(self):
        """Free up memory by clearing models"""
        self._detector_model = None
        self._face_detector = None
        self._ocr_reader = None
        self._clip_model = None
        self._clip_preprocess = None
        logger.info("Models cleared from memory")
