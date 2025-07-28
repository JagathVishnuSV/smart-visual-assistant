import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect_dominant_color(image, k=1):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img)

    dominant = kmeans.cluster_centers_[0].astype(int)
    color = tuple(dominant)

    # Draw a color box
    height, width = 100, 100
    color_box = np.zeros((height, width, 3), np.uint8)
    color_box[:, :] = color[::-1]  # Convert RGB to BGR
    image[0:100, 0:100] = color_box

    return image, f"RGB: {color}"
