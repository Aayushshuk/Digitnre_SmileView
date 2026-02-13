import cv2
import os
import gradio as gr

from face_detection import detect_face_bbox
from face_landmarks import LandmarkDetector
from teeth_segmentation import teeth_mask_from_landmarks
from smile_transform import smile_whitening_transform

detector = LandmarkDetector()

def process(img_rgb):
    if img_rgb is None:
        return None, None, "No input image"
    
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    before = bgr.copy()

    bbox = detect_face_bbox(bgr)
    if bbox is None:
        out = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
        return out, out, "No face detected"
    
    lm = detector.detects(bgr)
    if lm is None:
        out=cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
        return out, out, "No landmarsk detected"
    
    teeth_mask, _= teeth_mask_from_landmarks(bgr, lm)
    after = smile_whitening_transform(bgr, teeth_mask)

    before_rgb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)
    return before_rgb , after_rgb, "OK"

demo = gr.Interface(fn=process,inputs =gr.Image(
        sources=["upload", "webcam"],
        type ="numpy",
        label="Upload an image or use webcam"
        ),
        outputs=[
            gr.Image(label="BEFORE (Original)"),
            gr.Image(label ="AFTER (AI Enhanced)"),
            gr.Textbox(label="Status")
        ],
        title ="SmileView Simulation Demo",
        description = "SmileView Before/After preview using face detection, landmark detection, teeth segmentation and smile transform")
PORT = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0",server_port=PORT)
