import cv2 
import numpy as np 

class Stream():
    def __init__(self, camera_src, detector, counter, buzz, socketio, classes):
        self.camera_src = camera_src
        self.camera = None
        self.prev_messages = ""
        self.socketio = socketio
        self.detector = detector
        self.counter = counter
        self.buzz = buzz
        self.classes = classes
        
    def gen_frames(self):
        self.prev_messages = ""  
        while True:
            if self.camera is not None :
                ret, frame = self.camera.read()
                if not ret:
                    break

                frame  = self.detector.detect_object(frame)
                frame = self.counter.draw_line(frame)

                messages = []
                for counter_object in self.counter.counter_objects:
                    msg = {}
                    for key in counter_object :
                        if counter_object[key]['counter'] > 0 :
                            msg[self.classes[key]] = counter_object[key]['counter']
                    messages.append(msg)
                    
                if len(messages) > 0 :
                    # trigger buzzer & send counter to browser
                    if messages != self.prev_messages :
                        self.buzz.is_running = True
                        self.prev_messages = messages
                        self.socketio.emit("counter_event", {
                                                        "type" : self.counter.counter_mode,
                                                        "messages" : messages
                                                        })

                ret, buffer = cv2.imencode('.jpg', frame)
                if frame is None :
                    continue
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