# framework to simulate picamera2 for testing on non-camera systems
# currently only supports static image
import os
import numpy as np  
import cv2
from config import *

class DummyRequest:
    def __init__(self, arr):
        self._arr = arr

    def make_array(self, key):
        return self._arr

    def release(self):
        return
    
class DummyPicamera2:
    def __init__(self):
        self._running = False
        self.test_img = None
        # try to load a test image from the script directory
        try:
            base = os.path.dirname(__file__)
            path = os.path.join(base, 'preview/test_img02.jpg')
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.test_img = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
                    print(f"Loaded test image from {path} for DummyPicamera2.")
        except Exception:
            self.test_img = None

    def create_video_configuration(self, *args, **kwargs):
        return {}

    def configure(self, cfg):
        return

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def set_controls(self, controls):
        return

    def capture_request(self):
        # If a test image is available, return it (keeps static preview)
        if self.test_img is not None:
            return DummyRequest(self.test_img.copy())

        # otherwise produce a  so preview works on non-camera systems
        arr = np.full((CAM_H, CAM_W), 128, dtype=np.uint8)
        return DummyRequest(arr)