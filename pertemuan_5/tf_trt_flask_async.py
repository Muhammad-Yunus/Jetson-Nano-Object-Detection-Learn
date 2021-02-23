from flask import Flask, render_template, Response
from flask_socketio import SocketIO

import cv2
import json 
import datetime
import numpy as np

from tf_trt_detector import Detector
from utils import Utils

detector = Detector()
utils = Utils()

# load label map
classesFile = "coco.json"
with open(classesFile) as json_labels:
    classes = json.load(json_labels)

# generate random color
color_maps = {}
for key in classes :
    color_maps[key] = (np.random.randint(0,255), 
                       np.random.randint(0,255), 
                       np.random.randint(0,255))

# load TF-TRT Optimized Inference Graph (.pb)
# "ssd_inception_v2_coco_trt.pb"
# "faster_rcnn_inception_v2_coco_trt.pb"
MODEL_NAME = "ssd_inception_v2_coco_trt.pb"
BASE_PATH = "tf-trt-graph/"

detector.load_tf_graph(BASE_PATH + MODEL_NAME)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'
socketio = SocketIO(app)

camera = cv2.VideoCapture(2)

class FRCNN():
    def __init__(self):
        self.output = []
        self.frame = []

        # parameter
        self.target_w = 300
        self.target_h = 300

    def main(self):
        while True :
            if self.frame != [] :
                frame_c = self.frame.copy()
                frame_c = cv2.resize(frame_c, (self.target_w , self.target_h)) 

                # predict classess & box
                a = datetime.datetime.now()
                self.output = detector.detect(frame_c)
                b = datetime.datetime.now()
                c = b - a
                print('Inference time: %.2fs' % (c.microseconds/1000000))

    def detect_object(self, frame):
        self.frame = frame

        if self.output != [] :
            return utils.postprocess(self.output, frame, classes, font_size=0.8, color_maps=color_maps)
        else :
            return frame

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = frame[:, 80:-80]
            #frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame  = frcnn.detect_object(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if frame is None :
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    global frcnn 
    frcnn = FRCNN()
    socketio.start_background_task(target=frcnn.main)
    app.run(host="0.0.0.0")