from flask import Flask, render_template, Response
from flask_socketio import SocketIO

import cv2
import json 
import numpy as np

from utils import Utils
from gst_cam import camera
from stitching_utils import crop_border

utils = Utils()

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

# load petrained model (.pb & .pbtxt)
net = cv2.dnn.readNetFromTensorflow("model/frozen_inference_graph.pb",
                                    "model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

# set CUDA as backend & target OpenCV DNN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get output layers
layerOutput = net.getUnconnectedOutLayersNames()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'
socketio = SocketIO(app)

StitcObj = cv2.Stitcher(try_use_gpu=True)
stitcher = StitcObj.create(cv2.Stitcher_PANORAMA)

w, h = 480, 320
cap_0 = cv2.VideoCapture(camera(0, w, h))
cap_1 = cv2.VideoCapture(camera(1, w, h))

class SSD():
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
        ret_0, frame_0 = cap_0.read()
        if not ret_0:
            break
        ret_1, frame_1 = cap_1.read()
        if not ret_1:
            break

        try :
            status, frame = stitcher.stitch([frame_0, frame_1])
            if status != cv2.Stitcher_OK:
                print("Can't stitch images, error code = %d" % status)
                frame = np.hstack((frame_0, frame_1))
            else :
                frame = crop_border(frame)
        except Exception as e:
            print(e)
            continue

        frame  = ssd.detect_object(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if frame is None :
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', w=2*w)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    global ssd 
    ssd = SSD()
    socketio.start_background_task(target=ssd.main)
    app.run(host="0.0.0.0")