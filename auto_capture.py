import cv2
import numpy as np
import time

class AutoCapture:


    def __init__(self, stable_seconds=1.2, move_tol_px=1.8, scale_tol=0.12, teeth_px_thr=350):
        self.stable_seconds = stable_seconds
        self.move_tol_px=move_tol_px
        self.scale_tol = scale_tol
        self.teeth_px_thr = teeth_px_thr

        self.prev_bbox = None
        self.stable_start = None
        self.last_capture_time =0

    def _bbox_ok(self, bbox):
        return bbox is not None and bbox[2] > 30 and bbox[3] > 30
    
    def _bbox_stable(self, bbox):
        if self.prev_bbox is None:
            self.prev_bbox = bbox
            self.stable_start = time.time()
            return False
        
        x,y,w,h = bbox
        px,py,pw,ph = self.prev_bbox

        cx, cy = x + w/2.0, y+h/2.0
        pcx, pcy = px + pw/2.0, py + ph/2.0
        move = np.hypot(cx - pcx, cy - pcy)

        scale = abs((w*h) - (pw*ph))/max(1.0,(pw*ph))

        if move <= self.move_tol_px and scale <= self.scale_tol:

            if self.stable_start is None:
                self.stable_start = time.time()
            return (time.time() - self.stable_start) >= self.stable_seconds
        else:

            self.prev_bbox = bbox
            self.stable_start = time.time()
            return False
        
    def teeth_visible(self, teeth_mask):
        if teeth_mask is None:
            return False
        return int((teeth_mask > 0).sum()) >= self.teeth_px_thr
    
    def should_capture(self, quality_ok, bbox, teeth_mask):
        if time.time() - self.last_capture_time < 2.0:
            return False
        if not quality_ok:
            return False
        if not self._bbox_ok(bbox):
            return False
        if not self.teeth_visible(teeth_mask):
            return False
        
        return self._bbox_stable(bbox)
    
    def mark_capture(self):
        self.last_capture_time = time.time()
        self.prev_bbox = None
        self.stable_start = None

def save_before_after(before_bgr, after_bgr, out_dir="captures"):
    import  os
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
        
    before_path = os.path.join(out_dir, f"before_{ts}.jpg")
    after_path  = os.path.join(out_dir, f"after_{ts}.jpg")

    cv2.imwrite(before_path, before_bgr)
    cv2.imwrite(after_path, after_bgr)

    return before_path, after_path