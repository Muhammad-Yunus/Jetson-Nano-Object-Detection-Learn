from flask import Flask, render_template, Response
import numpy as np
import cv2

from gst_cam import camera

app = Flask(__name__)

w, h = 480, 320
cap_0 = cv2.VideoCapture(camera(0, w, h))
cap_1 = cv2.VideoCapture(camera(1, w, h))

def gen_frames():  
    while True:
        ret_0, frame_0 = cap_0.read()
        if not ret_0:
            break
        ret_1, frame_1 = cap_1.read()
        if not ret_1:
            break

        frame = np.hstack((frame_0, frame_1))

        ret, buffer = cv2.imencode('.jpg', frame)
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

app.run(host="0.0.0.0")
cap_0.release()
cap_1.release()
