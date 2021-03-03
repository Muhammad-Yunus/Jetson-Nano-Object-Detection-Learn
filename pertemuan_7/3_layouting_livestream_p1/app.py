from flask import Flask, render_template, Response, request, flash

import numpy as np
import cv2

from core_service.stream import Stream
from core_service.gst_cam import camera

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'

w, h = 480, 320
stream = Stream(camera_src=camera(0, w, h))

@app.route("/")
def index():
    camera = request.args.get("camera")

    if camera is not None and camera == 'off' and stream.status() == True:
        stream.close()
        flash("Camera turn off!", "info")
    elif camera is not None and camera == 'on' and stream.status() == False:
        stream.open()
        flash("Camera turn on!", "success")

    setting = dict(
        stream_on = stream.status(),
        w = w
    )
    return render_template("index.html", setting = setting)

@app.route('/video_feed')
def video_feed():
    return Response(stream.gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/setting")
def setting():
    return render_template("setting.html")
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)