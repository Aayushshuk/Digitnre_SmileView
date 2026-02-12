import cv2 
import numpy as np

def poisson_blend(before_bgr, after_bgr, mask_255):

    mask = (mask_255 > 0).astype(np.uint8) * 255
    ys,xs = np.where(mask > 0)
    if len(xs) == 0:
        return before_bgr
    
    center = (int(xs.mean()), int(ys.mean()))
    blended = cv2.seamlessClone(after_bgr, before_bgr, mask, center, cv2.NORMAL_CLONE)
    return blended