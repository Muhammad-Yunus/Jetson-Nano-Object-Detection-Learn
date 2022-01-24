from flask import Flask, render_template, Response

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

from gst_cam import camera
from utils import Drawer, PostprocessYOLO, PreprocessYOLO
import yolov3_onnx.common as common

TRT_LOGGER = trt.Logger()

class TRT_Detector():
    def __init__(self, engine_file_path):
        self.drawer = Drawer()
        self.engine_file_path = engine_file_path
        self.output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]

        print("[INFO] Reading engine from file %s ..." % self.engine_file_path)
        with open(self.engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def detect(self, img):
        img_raw = img.copy()
        print("[INFO] preprocessing..")
        img = preprocessor.process(img)
        shape_orig_WH = img_raw.shape[:2][::-1]

        print("[INFO] do inference..")
        trt_outputs = []
        with self.engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
            inputs[0].host = img
            trt_outputs = common.do_inference_v2(context, 
                                                bindings=bindings, 
                                                inputs=inputs, 
                                                outputs=outputs, 
                                                stream=stream)
        print("[INFO] postprocessing..")
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
        boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))

        try :
            print(len(boxes), len(classes), len(scores))
            img_raw = self.drawer.draw(img_raw, boxes, scores, classes, all_categories)
        except Exception as e:
            print("[ERROR] ", e)

        return img_raw

def load_label_categories(label_file_path):
    categories = [line.rstrip('\n') for line in open(label_file_path)]
    return categories

all_categories = load_label_categories('model/coco_labels.txt')

input_size = (416, 416)
preprocessor = PreprocessYOLO(input_size)

postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                  
                    "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  
                                    (59, 119), (116, 90), (156, 198), (373, 326)],
                    "obj_threshold": 0.6,                                               
                    "nms_threshold": 0.5,                                               
                    "yolo_input_resolution": input_size,
                    "category_num": len(all_categories)}
postprocessor = PostprocessYOLO(**postprocessor_args)

engine_file_path = "model/yolov3.trt"
trt_detector = TRT_Detector(engine_file_path)

app = Flask(__name__)

w, h = 480, 320
cap_0 = cv2.VideoCapture(camera(0, w, h))

def gen_frames():  
    while True:
        ret_0, frame_0 = cap_0.read()
        if not ret_0:
            break

        frame_0 = frame_0[:, 80:-80]

        frame_0 = trt_detector.detect(frame_0)

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

app.run(host="0.0.0.0", threaded=False)
cap_0.release()
