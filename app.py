import os
import cv2
import gradio as gr

from face_detection import detect_face_bbox
from face_landmarks import LandmarkDetector
from teeth_segmentation import teeth_mask_from_landmarks
from smile_transform import smile_whitening_transform

# Safer: lazy-init (so deployment reloads won't break)
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = LandmarkDetector()
    return _detector

def process(img_rgb):
    # Gradio returns RGB numpy array (H,W,3)
    if img_rgb is None:
        return None, None, "No input image"

    # Convert to BGR for OpenCV processing
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    before_bgr = bgr.copy()

    # Always create BEFORE for UI
    before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)

    # Face bbox
    bbox = detect_face_bbox(bgr)
    if bbox is None:
        return before_rgb, before_rgb, "No face detected"

    # Landmarks (MediaPipe Tasks)
    detector = get_detector()
    lm = detector.detects(bgr)
    if lm is None:
        return before_rgb, before_rgb, "No landmarks detected"

    # Teeth segmentation mask
    teeth_mask, _ = teeth_mask_from_landmarks(bgr, lm)
    if teeth_mask is None:
        return before_rgb, before_rgb, "Teeth mask not found"

    # Smile transform (whitening)
    after_bgr = smile_whitening_transform(bgr, teeth_mask)
    after_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)

    return before_rgb, after_rgb, "OK"

demo = gr.Interface(
    fn=process,
    inputs=gr.Image(
        sources=["upload", "webcam"],
        type="numpy",
        label="Upload an image or use webcam"
    ),
    outputs=[
        gr.Image(label="BEFORE (Original)"),
        gr.Image(label="AFTER (AI Enhanced)"),
        gr.Textbox(label="Status")
    ],
    title="SmileView Simulation Demo",
    description="SmileView Before/After preview using face detection, landmark detection, teeth segmentation and smile transform"
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 8080)),
    show_error=True
)
