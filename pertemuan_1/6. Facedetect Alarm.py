from flask import Flask, render_template, Response
import Jetson.GPIO as GPIO
import time
import cv2

output_pin = 7
GPIO.setmode(GPIO.BOARD)  
GPIO.setwarnings(False)
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def play_beep():
    print("play beep...")
    for _ in range(3):
        GPIO.output(output_pin, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(output_pin, GPIO.LOW)
        time.sleep(0.02)

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
        play_beep()

    return frame

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_face(frame)
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
    app.run(host="0.0.0.0")
    print("cleanup..")
    GPIO.cleanup()