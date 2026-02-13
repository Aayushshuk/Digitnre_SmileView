import os
import urllib.request
import numpy as np
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Downloaded:", MODEL_PATH)

class LandmarkDetector:
    def __init__(self):
        _ensure_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detects(self, frame_bgr):
        # MediaPipe Tasks expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None

        # return list of landmarks similar to your old code (with .x .y)
        return result.face_landmarks[0]

    def close(self):
        self.detector.close()
