import cv2
import numpy as np

class Detector():
    def __init__(self, counter, socketio, classes):
        self.classes = classes
        self.color_maps = {}
        self.net = None
        self.layerOutput = []

        self.output = []
        self.frame = []

        self.target_w = 244
        self.target_h = 244

        self.counter = counter
        self.socketio = socketio

    def generate_color_maps(self):
        self.color_maps = {}
        for key in self.classes :
            self.color_maps[key] = (np.random.randint(0,255), 
                            np.random.randint(0,255), 
                            np.random.randint(0,255))

    def load_model(self, model="model/frozen_inference_graph.pb",
                        config="model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"):
        # load pretrained model
        self.net = cv2.dnn.readNet(model, config)

        # set CUDA as backend & target OpenCV DNN
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # get output layers
        self.layerOutput = self.net.getUnconnectedOutLayersNames()

    def main(self):
        while True :
            if self.frame != [] :
                blob = cv2.dnn.blobFromImage(self.frame, 1.0, 
                                            (self.target_w, self.target_h), 
                                            (0, 0, 0), swapRB=True, crop=False)

                # predict classess & box
                self.net.setInput(blob)
                self.output = self.net.forward(self.layerOutput)
                
                t, _ = self.net.getPerfProfile()
                inf = '%.2f' % (1.0/(t / cv2.getTickFrequency()))
                self.socketio.emit("inference_event", inf)

    def detect_object(self, frame):
        self.frame = frame
        if self.output != [] :
            return self.counter.postprocess(self.output, frame, font_size=0.5, color_maps=self.color_maps)
        else :
            return frame