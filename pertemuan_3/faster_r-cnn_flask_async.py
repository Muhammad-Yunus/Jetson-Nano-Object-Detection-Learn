from flask import Flask, render_template, Response
from flask_socketio import SocketIO

import cv2
import json 
import numpy as np

from utils import Utils

utils = Utils()

classesFile = "object-detection.json"
with open(classesFile) as json_labels:
    classes = json.load(json_labels)


# load petrained model (.pb & .pbtxt) faster R-CNN with backbone Inception V2
net = cv2.dnn.readNetFromTensorflow("model/frozen_inference_graph.pb",
                                    "model/faster_rcnn_inception_v2_custom_dataset.pbtxt")

# set CUDA as backend & target OpenCV DNN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get output layers
layerOutput = net.getUnconnectedOutLayersNames()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'
socketio = SocketIO(app)

camera = cv2.VideoCapture(0)

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
        # self.target_w = frame.shape[1]
        # self.target_h = frame.shape[0]
        if self.output != [] :
            return utils.postprocess(self.output, frame, classes, font_size=0.8)
        else :
            return frame

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = frame[:, 80:-80]
            frame  = frcnn.detect_object(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
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