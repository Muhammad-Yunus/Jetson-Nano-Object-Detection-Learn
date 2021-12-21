from flask import Flask, render_template, Response
from flask_socketio import SocketIO

import cv2
import json 
import datetime
import numpy as np

from utils import Utils
utils = Utils()

classesFile = "object-detection.json"
# classesFile = "coco.json"
with open(classesFile) as json_labels :
    classes = json.load(json_labels)

# load petrained model (.pb & .pbtxt) faster R-CNN with backbone Inception V2
modelConfiguration = "model/yolov3-tiny.cfg"
modelWeights = "model/yolov3-tiny-custom.weights"
# modelConfiguration = "model/coco_yolov3-tiny copy.cfg"
# modelWeights = "model/coco_yolov3-tiny copy.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# set CUDA as backend & target OpenCV DNN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get output layers
layerOutput = net.getUnconnectedOutLayersNames()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'
socketio = SocketIO(app)

camera = cv2.VideoCapture(0)

color_maps = {}
for key in classes :
    color_maps[key] = (np.random.randint(0,255), 
                       np.random.randint(0,255), 
                       np.random.randint(0,255))

class YoloInference():
    def __init__(self):
        self.output = []
        self.frame = []

        # parameter
        self.target_w = 416
        self.target_h = 416

    def main(self):
        while True :
            if self.frame != [] :
                blob = cv2.dnn.blobFromImage(
                            self.frame, 
                            1/255, 
                            (self.target_w, self.target_h), 
                            [0, 0, 0],
                            1, 
                            crop=False)

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
        a = datetime.datetime.now()
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = frame[:, 80:-80]

            frame  = frcnn.detect_object(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            b = datetime.datetime.now()
            c = b - a
            #print('Load time: %.2fs' % (c.microseconds/1000000))
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
    frcnn = YoloInference()
    socketio.start_background_task(target=frcnn.main)
    socketio.run(app, host="0.0.0.0")
