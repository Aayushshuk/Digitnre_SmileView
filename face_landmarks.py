import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join("models", "face_landmarker.task")

def _ensure_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

_ensure_model()

_base = python.BaseOptions(model_asset_path=MODEL_PATH)
_options = vision.FaceLandmarkerOptions(
    base_options=_base,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
_landmarker = vision.FaceLandmarker.create_from_options(_options)

class LandmarkDetector:
    def detects(self, bgr_img):
        h, w = bgr_img.shape[:2]
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = _landmarker.detect(image)
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]  # list of normalized landmarks
        # return pixel coords [(x,y), ...]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        return pts
