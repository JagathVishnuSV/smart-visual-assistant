"""
Live Video Stream Module
Real-time video analysis with continuous environment description
"""
import cv2
import numpy as np
import logging
import threading
import time
from typing import Callable, Optional
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveVideoStream:
    """Manages live video streaming and real-time analysis"""
    
    def __init__(self, camera_index=0, fps=10, width=640, height=480):
        """
        Initialize live video stream
        
        Args:
            camera_index: Camera device index
            fps: Target frames per second for analysis
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index
        self.fps = fps
        self.width = width
        self.height = height
        
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.analysis_callback = None
        self.analysis_interval = 1.0 / fps
        self.last_analysis_time = 0
        
        self.stream_thread = None
        self.analysis_thread = None
        
    def start(self, analysis_callback: Optional[Callable] = None):
        """
        Start video stream
        
        Args:
            analysis_callback: Function to call with each frame for analysis
        """
        if self.is_running:
            logger.warning("Stream already running")
            return False
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            self.analysis_callback = analysis_callback
            
            # Start streaming thread
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()
            
            # Start analysis thread if callback provided
            if analysis_callback:
                self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
                self.analysis_thread.start()
            
            logger.info("Live video stream started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop video stream"""
        self.is_running = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Live video stream stopped")
    
    def _stream_loop(self):
        """Main streaming loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Small delay to control frame rate
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in stream loop: {e}")
                time.sleep(0.1)
    
    def _analysis_loop(self):
        """Analysis loop that calls callback at specified intervals"""
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_analysis_time >= self.analysis_interval:
                    frame = self.get_current_frame()
                    
                    if frame is not None and self.analysis_callback:
                        try:
                            self.analysis_callback(frame)
                        except Exception as e:
                            logger.error(f"Error in analysis callback: {e}")
                    
                    self.last_analysis_time = current_time
                
                time.sleep(0.05)  # Check frequently
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(0.1)
    
    def get_current_frame(self):
        """Get the current frame safely"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_frame_generator(self):
        """Generator for streaming frames"""
        while self.is_running:
            frame = self.get_current_frame()
            if frame is not None:
                yield frame
            time.sleep(0.033)  # ~30 fps
    
    def set_analysis_interval(self, seconds):
        """Set how often to run analysis (in seconds)"""
        self.analysis_interval = seconds
    
    def update_analysis_callback(self, callback: Callable):
        """Update the analysis callback function"""
        self.analysis_callback = callback


class ContinuousSceneDescriber:
    """Continuously analyzes and describes the environment"""
    
    def __init__(self, vlm_model, speak_func, min_description_interval=5.0):
        """
        Initialize continuous scene describer
        
        Args:
            vlm_model: Vision-language model for analysis
            speak_func: Function to speak descriptions
            min_description_interval: Minimum seconds between descriptions
        """
        self.vlm_model = vlm_model
        self.speak_func = speak_func
        self.min_description_interval = min_description_interval
        
        self.last_description = ""
        self.last_description_time = 0
        self.last_objects = set()
        
    def analyze_and_describe(self, frame):
        """
        Analyze frame and speak description if scene changed significantly
        
        Args:
            frame: Video frame to analyze
        """
        try:
            current_time = time.time()
            
            # Check if enough time has passed
            if current_time - self.last_description_time < self.min_description_interval:
                return
            
            # Detect objects
            _, objects = self.vlm_model.detect_objects(frame)
            objects_set = set(objects)
            
            # Check if scene changed significantly
            if not self._scene_changed_significantly(objects_set):
                return
            
            # Create description
            if objects:
                if len(objects) == 1:
                    description = f"I see a {objects[0]}"
                elif len(objects) == 2:
                    description = f"I see a {objects[0]} and a {objects[1]}"
                else:
                    description = f"I see {len(objects)} objects: {', '.join(objects[:3])}"
                    if len(objects) > 3:
                        description += f" and {len(objects) - 3} more"
            else:
                description = "No specific objects detected in view"
            
            # Only speak if description is different
            if description != self.last_description:
                self.speak_func(description)
                self.last_description = description
                self.last_description_time = current_time
                self.last_objects = objects_set
                
        except Exception as e:
            logger.error(f"Error in scene description: {e}")
    
    def _scene_changed_significantly(self, new_objects):
        """Check if scene changed enough to warrant new description"""
        if not self.last_objects:
            return True
        
        # Calculate similarity
        intersection = len(self.last_objects.intersection(new_objects))
        union = len(self.last_objects.union(new_objects))
        
        if union == 0:
            return False
        
        similarity = intersection / union
        
        # If similarity is less than 60%, scene changed significantly
        return similarity < 0.6
    
    def reset(self):
        """Reset description history"""
        self.last_description = ""
        self.last_objects = set()
        self.last_description_time = 0
