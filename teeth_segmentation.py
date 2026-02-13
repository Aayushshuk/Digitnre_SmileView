import cv2
import numpy as np

MOUTH_LEFT  = 61
MOUTH_RIGHT = 291
UPPER_LIP   = 13
LOWER_LIP   = 14

def _xy(point):
    """
    Supports:
    - MediaPipe landmark objects: point.x, point.y (normalized 0..1)
    - Tuples/lists: (x, y) where x,y are either normalized (0..1) OR pixels
    """
    # tuple/list -> (x, y)
    if isinstance(point, (tuple, list)):
        return float(point[0]), float(point[1])
    # mediapipe landmark object
    return float(point.x), float(point.y)

def _pt(lm, idx, w, h):
    x, y = _xy(lm[idx])

    # If landmarks are already in pixel coordinates, don't multiply by w/h
    # Heuristic: if x>2 or y>2 it's almost surely pixels (since normalized is <=1)
    if x > 2 or y > 2:
        return int(x), int(y)

    # normalized -> pixel
    return int(x * w), int(y * h)

def teeth_mask_from_landmarks(frame_bgr, lm):
    h, w = frame_bgr.shape[:2]

    ml = _pt(lm, MOUTH_LEFT,  w, h)
    mr = _pt(lm, MOUTH_RIGHT, w, h)
    up = _pt(lm, UPPER_LIP,   w, h)
    lo = _pt(lm, LOWER_LIP,   w, h)

    x1 = max(0, min(ml[0], mr[0]) - 25)
    x2 = min(w, max(ml[0], mr[0]) + 25)
    y1 = max(0, min(up[1], lo[1]) - 25)
    y2 = min(h, max(up[1], lo[1]) + 45)

    mask = np.zeros((h, w), dtype=np.uint8)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return mask, (x1, y1, x2, y2)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    L = cv2.equalizeHist(L)
    _, thr = cv2.threshold(L, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask[y1:y2, x1:x2] = thr
    return mask, (x1, y1, x2, y2)
