import cv2
import mediapipe as mp

mp_fm = mp.solutions.face_mesh

class LandmarkDetector:
    def __init__(self):
        self.fm = mp_fm.FaceMesh(static_image_mode =False,
                                refine_landmarks =True,
                                max_num_faces=1)
        
    def detects(self, frame_bgr):

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        return res.multi_face_landmarks[0].landmark

    def close(self):
        self.fm.close()