import cv2
import numpy as np

def blur_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def quality_flags(bgr, blur_thr=35.0, bright_low=60, bright_high=200):
    bs = blur_score(bgr)
    br = brightness_score(bgr)
    blur_ok = bs >= blur_thr
    bright_ok = bright_low <= br <= bright_high
    return blur_ok, bright_ok, bs, br
