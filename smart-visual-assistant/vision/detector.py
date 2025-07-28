from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self,model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        #self.model.fuse()  # Fuse model layers for faster inference

    def detect(self,frame):
        results = self.model(frame)[0]  # Get the first result
        objects = [self.model.names[int(cls)] for cls in results.boxes.cls]
        return results.plot(), list(set(objects)) ## Frame with boxes, unique labels