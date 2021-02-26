from flask import Flask, render_template, Response
import numpy as np
import cv2

from gst_cam import camera

app = Flask(__name__)

w, h = 480, 360
cap_0 = cv2.VideoCapture(camera(0, w, h))

def gen_frames():  
    while True:
        ret_0, frame_0 = cap_0.read()
        if not ret_0:
            break

        ret, buffer = cv2.imencode('.jpg', frame_0)
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

app.run(host="0.0.0.0")
cap_0.release()
