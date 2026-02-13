import cv2

# OpenCV built-in face detector (works on servers too)
_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_bbox(frame_bgr):
    """
    Returns bbox as (x, y, w, h) or None
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None

    # pick largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return (int(x), int(y), int(w), int(h))
