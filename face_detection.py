import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = os.path.join("models", "blaze_face_short_range.tflite")

def _ensure_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

_ensure_model()

_base = python.BaseOptions(model_asset_path=MODEL_PATH)
_options = vision.FaceDetectorOptions(base_options=_base)
_detector = vision.FaceDetector.create_from_options(_options)

def detect_face_bbox(bgr_img):
    # bgr -> rgb
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = _detector.detect(image)
    if not result.detections:
        return None

    bb = result.detections[0].bounding_box  # pixel bbox
    x1, y1 = int(bb.origin_x), int(bb.origin_y)
    x2, y2 = int(bb.origin_x + bb.width), int(bb.origin_y + bb.height)
    return (x1, y1, x2, y2)
