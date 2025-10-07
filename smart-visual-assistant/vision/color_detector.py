"""
Color Detection Module
Detects dominant colors in images using K-means clustering
"""
import cv2
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_dominant_color(image: np.ndarray, k=1, sample_size=150) -> Tuple[np.ndarray, str]:
    """
    Detect dominant color in image
    
    Args:
        image: Input image (BGR format)
        k: Number of dominant colors to find
        sample_size: Resize image to this size for faster processing
    
    Returns:
        Tuple of (annotated image, color description)
    """
    try:
        from sklearn.cluster import KMeans
        
        # Downsample image for faster processing
        img_sample = cv2.resize(image, (sample_size, sample_size))
        img_rgb = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1, 3))
        
        # Find dominant color using K-means
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        dominant_rgb = kmeans.cluster_centers_[0].astype(int)
        color_rgb = tuple(dominant_rgb)
        color_bgr = color_rgb[::-1]  # Convert RGB to BGR for OpenCV
        
        # Draw color swatch on image
        swatch_size = 100
        cv2.rectangle(
            image, (10, 10), (10 + swatch_size, 10 + swatch_size),
            color_bgr, -1
        )
        cv2.rectangle(
            image, (10, 10), (10 + swatch_size, 10 + swatch_size),
            (0, 0, 0), 2
        )
        
        # Get color name
        color_name = get_color_name(color_rgb)
        
        # Add text label
        cv2.putText(
            image, color_name, (10, swatch_size + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            image, color_name, (10, swatch_size + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
        )
        
        return image, f"{color_name} (RGB: {color_rgb})"
        
    except Exception as e:
        logger.error(f"Color detection failed: {e}")
        return image, "Unknown"


def get_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Approximate color name from RGB values
    
    Args:
        rgb: RGB tuple (r, g, b)
    
    Returns:
        Approximate color name
    """
    r, g, b = rgb
    
    # Define color thresholds
    if r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif r < 100 and g < 100 and b < 100:
        return "Dark Gray"
    elif r > 150 and g > 150 and b > 150:
        return "Light Gray"
    
    # Primary and secondary colors
    if r > max(g, b) + 50:
        if g > 100:
            return "Orange" if g > b else "Red-Orange"
        return "Red"
    elif g > max(r, b) + 50:
        if r > 100:
            return "Yellow-Green" if r > b else "Green"
        return "Green"
    elif b > max(r, g) + 50:
        return "Blue"
    
    # Mixed colors
    if r > 150 and g > 150 and b < 100:
        return "Yellow"
    elif r > 150 and b > 150 and g < 100:
        return "Magenta"
    elif g > 150 and b > 150 and r < 100:
        return "Cyan"
    elif r > 100 and g > 100 and b > 100:
        return "Gray"
    
    return "Mixed Color"


def get_color_palette(image: np.ndarray, n_colors=5) -> Tuple[np.ndarray, list]:
    """
    Extract color palette from image
    
    Args:
        image: Input image
        n_colors: Number of colors in palette
    
    Returns:
        Tuple of (image with palette, list of RGB colors)
    """
    try:
        from sklearn.cluster import KMeans
        
        img_sample = cv2.resize(image, (150, 150))
        img_rgb = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1, 3))
        
        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        
        # Draw palette
        palette_height = 50
        palette_width = image.shape[1]
        segment_width = palette_width // n_colors
        
        for i, color in enumerate(colors):
            x1 = i * segment_width
            x2 = (i + 1) * segment_width
            color_bgr = tuple(map(int, color[::-1]))
            cv2.rectangle(
                image, (x1, 0), (x2, palette_height),
                color_bgr, -1
            )
        
        return image, [tuple(c) for c in colors]
        
    except Exception as e:
        logger.error(f"Palette extraction failed: {e}")
        return image, []
