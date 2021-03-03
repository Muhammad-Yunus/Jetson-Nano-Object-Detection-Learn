import cv2 
import numpy as np 

class Stream():
    def __init__(self, camera_src):
        self.camera_src = camera_src
        self.camera = None

    def gen_frames(self):  
        while True:
            if self.camera is not None :
                ret, frame = self.camera.read()
                if not ret:
                    break

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def close(self):
        if self.camera is not None :
            self.camera.release()
            self.camera = None

    def open(self):
        self.camera = cv2.VideoCapture(self.camera_src)

    def status(self):
        return self.camera is not None