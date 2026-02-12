import cv2
import mediapipe as mp

mp_fd = mp.solutions.face_detection

def detect_face_bbox(frame_bgr, min_conf=0.6):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=min_conf) as fd:
        res = fd.process(rgb)

    if not res.detections:
        return None

    det = res.detections[0]
    bb = det.location_data.relative_bounding_box
    x = int(max(0, bb.xmin * w))
    y = int(max(0, bb.ymin * h))
    bw = int(min(w - x, bb.width * w))
    bh = int(min(h - y, bb.height * h))
    return (x, y, bw, bh)
