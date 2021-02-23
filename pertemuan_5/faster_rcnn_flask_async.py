from flask import Flask, render_template, Response
from flask_socketio import SocketIO

import cv2
import json 
import numpy as np

from utils import Utils

utils = Utils()

# load label map
classesFile = "coco-ssd.json"
# "coco-frcnn.json"
# "coco-ssd.json"
with open(classesFile) as json_labels:
    classes = json.load(json_labels)

# generate random color
color_maps = {}
for key in classes :
    color_maps[key] = (np.random.randint(0,255), 
                       np.random.randint(0,255), 
                       np.random.randint(0,255))

# load petrained model (.pb & .pbtxt)
MODEL_NAME = "ssd_mobilenet_v2_coco_2018_03_29"
# "faster_rcnn_inception_v2_coco_2018_01_28"
# "faster_rcnn_resnet50_coco_2018_01_28"
# "ssd_mobilenet_v2_coco_2018_03_29"

net = cv2.dnn.readNetFromTensorflow("tf_models/%s/frozen_inference_graph.pb" % MODEL_NAME,
                                    "graph_text/%s.pbtxt" % MODEL_NAME)

# set CUDA as backend & target OpenCV DNN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get output layers
layerOutput = net.getUnconnectedOutLayersNames()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'
socketio = SocketIO(app)

camera = cv2.VideoCapture(2)

class FRCNN():
    def __init__(self):
        self.output = []
        self.frame = []

        # parameter
        self.target_w = 244
        self.target_h = 244

    def main(self):
        while True :
            if self.frame != [] :
                blob = cv2.dnn.blobFromImage(self.frame, 1.0, (self.target_w, self.target_h), (0, 0, 0), swapRB=True, crop=False)

                # predict classess & box
                net.setInput(blob)
                self.output = net.forward(layerOutput)
                
                t, _ = net.getPerfProfile()
                print('inference time: %.2f s' % (t / cv2.getTickFrequency()))

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