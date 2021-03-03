from flask import Flask, render_template, Response
from flask_socketio import SocketIO

import cv2
import json 
import numpy as np

from gst_cam import camera
from object_counter import Counter
from buzzer import Buzzer

# load label map
classesFile = "model/coco-ssd.json"
with open(classesFile) as json_labels:
    classes = json.load(json_labels)

# generate random color
color_maps = {}
for key in classes :
    color_maps[key] = (np.random.randint(0,255), 
                       np.random.randint(0,255), 
                       np.random.randint(0,255))

# load pretrained model
net = cv2.dnn.readNet("model/frozen_inference_graph.pb",
                    "model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

# set CUDA as backend & target OpenCV DNN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get output layers
layerOutput = net.getUnconnectedOutLayersNames()

# initialize flask & socketio
app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'
socketio = SocketIO(app)

# initialize camera object
w, h = 480, 360
cap_0 = cv2.VideoCapture(camera(0, w, h)) # if using CSI camera
#cap_0 = cv2.VideoCapture(0) # if using USB camera

# initialize counter object
lines = []
lines.append([int(w*0.20), 0, int(w*0.20), h]) # LINE 0, x0, y0, x1, y1
lines.append([int(w*0.80), 0, int(w*0.80), h]) # LINE 1, x0, y0, x1, y1
counter = Counter(classes, mode='multiline', lines=lines, threshDist = 30) #  mode='line', 'area', 'multiline'


class Detector():
    def __init__(self):
        self.output = []
        self.frame = []

        # parameter
        self.target_w = 244
        self.target_h = 244

    def main(self):
        while True :
            if self.frame != [] :
                blob = cv2.dnn.blobFromImage(self.frame, 1.0, 
                                            (self.target_w, self.target_h), 
                                            (0, 0, 0), swapRB=True, crop=False)

                # predict classess & box
                net.setInput(blob)
                self.output = net.forward(layerOutput)
                
                t, _ = net.getPerfProfile()
                inf = 'inference time: %.2f fps' % (1.0/(t / cv2.getTickFrequency()))
                socketio.emit("inference_event", inf)

    def detect_object(self, frame):
        self.frame = frame
        if self.output != [] :
            return counter.postprocess(self.output, frame, font_size=0.5, color_maps=color_maps)
        else :
            return frame

def gen_frames(): 
    prev_messages = [] 
    while True:
        success, frame = cap_0.read()
        if not success:
            break
        else:
            frame  = detector.detect_object(frame)
            frame = counter.draw_line(frame)

            messages = []
            for counter_object in counter.counter_objects:
                msg = {}
                for key in counter_object :
                    if counter_object[key]['counter'] > 0 :
                        msg[classes[key]] = counter_object[key]['counter']
                messages.append(msg)
                
            if len(messages) > 0 :
                # trigger buzzer & send counter to browser
                if messages != prev_messages :
                    buzz.is_running = True
                    prev_messages = messages
                    socketio.emit("counter_event", {
                                                    "type" : counter.counter_mode,
                                                    "messages" : messages
                                                    })

            ret, buffer = cv2.imencode('.jpg', frame)
            if frame is None :
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', w=w)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    global detector, buzz 
    buzz = Buzzer()
    detector = Detector()
    socketio.start_background_task(target=detector.main)
    socketio.start_background_task(target=buzz.main)
    app.run(host="0.0.0.0")
    cap_0.release()