"""
CLIP-based Visual Question Answering
Uses OpenAI's CLIP model for zero-shot image understanding
Optimized for CPU usage on low-compute systems
"""
import torch
import cv2
import numpy as np
import logging
from PIL import Image
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances (lazy loaded)
_model = None
_preprocess = None
_device = "cpu"  # Force CPU for low-compute compatibility


def get_clip_model():
    """Lazy load CLIP model"""
    global _model, _preprocess
    if _model is None:
        try:
            import clip
            logger.info("Loading CLIP model (this may take a moment)...")
            _model, _preprocess = clip.load("ViT-B/32", device=_device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise
    return _model, _preprocess


def answer_question(image_np: np.ndarray, question: str = None) -> Tuple[np.ndarray, str]:
    """
    Answer questions about image using CLIP
    
    Args:
        image_np: Input image (BGR numpy array)
        question: Optional question (not directly used, but guides candidate selection)
    
    Returns:
        Tuple of (original image, answer text)
    """
    try:
        model, preprocess = get_clip_model()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess image
        image_input = preprocess(pil_image).unsqueeze(0).to(_device)
        
        # Define comprehensive candidate list
        candidates = [
            "person", "people", "crowd", "face", "man", "woman", "child",
            "cup", "mug", "glass", "bottle", "phone", "smartphone", "laptop",
            "computer", "monitor", "keyboard", "mouse", "tablet",
            "book", "notebook", "pen", "pencil", "paper",
            "table", "desk", "chair", "sofa", "bed",
            "door", "window", "wall", "floor",
            "car", "bicycle", "motorcycle", "bus", "truck",
            "dog", "cat", "bird", "animal",
            "food", "fruit", "vegetable", "plate", "bowl",
            "tree", "plant", "flower", "grass",
            "bag", "backpack", "suitcase", "box",
            "clock", "watch", "calendar",
            "light", "lamp", "candle",
            "nothing", "empty room", "outdoor scene", "indoor scene"
        ]
        
        # Create text prompts
        texts = [f"a photo of a {item}" for item in candidates]
        text_tokens = torch.clip.tokenize(texts).to(_device)
        
        # Calculate similarity
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
        
        # Get top predictions with confidence
        predictions = []
        for i in range(min(3, len(indices))):
            idx = indices[i]
            conf = values[i].item()
            if conf > 0.05:  # Only include if confidence > 5%
                predictions.append(candidates[idx])
        
        if predictions:
            if len(predictions) == 1:
                answer = f"I see: {predictions[0]}"
            elif len(predictions) == 2:
                answer = f"I see: {predictions[0]} and possibly {predictions[1]}"
            else:
                answer = f"I see: {predictions[0]}, possibly {predictions[1]} or {predictions[2]}"
        else:
            answer = "I'm not sure what I'm seeing in this image"
        
        return image_np, answer
        
    except Exception as e:
        logger.error(f"CLIP Q&A failed: {e}")
        return image_np, "Could not analyze the image"


def classify_scene(image_np: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Classify the overall scene type
    
    Args:
        image_np: Input image
    
    Returns:
        Tuple of (image, scene classification)
    """
    try:
        model, preprocess = get_clip_model()
        
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_input = preprocess(pil_image).unsqueeze(0).to(_device)
        
        # Scene categories
        scenes = [
            "indoor office", "indoor home", "outdoor street",
            "outdoor nature", "indoor restaurant", "indoor store",
            "outdoor park", "indoor classroom", "outdoor parking lot",
            "indoor bedroom", "indoor kitchen", "indoor bathroom"
        ]
        
        texts = [f"a photo of {scene}" for scene in scenes]
        text_tokens = torch.clip.tokenize(texts).to(_device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            best_idx = similarity.argmax().item()
        
        scene = scenes[best_idx]
        return image_np, f"Scene: {scene}"
        
    except Exception as e:
        logger.error(f"Scene classification failed: {e}")
        return image_np, "Could not classify scene"
