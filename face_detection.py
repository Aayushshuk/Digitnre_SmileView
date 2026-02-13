import cv2

# Uses OpenCV built-in haarcascade file
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face_bbox(bgr):
    """
    Returns (x, y, w, h) or None
    """
    if bgr is None:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # pick biggest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return (int(x), int(y), int(w), int(h))
