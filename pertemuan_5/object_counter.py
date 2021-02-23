import cv2
import numpy as np
from utils import Utils

utils = Utils()

class Counter():
    def __init__(self, classes, mode='area', threshDist = 50, direction=('left', 'right')):
        self.classes = classes
        self.frame_id = 0
        self.counter_object = {}
        self.counter_mode = mode
        self.line = [] 
        self.threshDist = threshDist
        self.direction = direction
        self.set_null_counter()

    def set_null_counter(self):
        for class_id in self.classes:
            self.counter_object[class_id] = {
                                        'in' : False,
                                        'frame_id' : 0,
                                        'counter' : 0     
                                        }        

    def counter_area(self, class_id):
        if self.counter_object[class_id]['frame_id'] == self.frame_id :
            self.counter_object[class_id]['counter'] += 1
        else :
            self.counter_object[class_id]['counter'] = 1
            self.counter_object[class_id]['frame_id'] = self.frame_id

    def counter_line_cross(self, class_id, x, y):
        dx = x - self.line[0]
        dy = y - self.line[1]
        r = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        if r < self.threshDist :
            if self.counter_object[class_id]['in'] == False :
                self.counter_object[class_id]['counter'] += 1
            self.counter_object[class_id]['in'] = True 
        else :
            self.counter_object[class_id]['in'] = False

    def postprocess(self, outs, frame,  
                    confThreshold = 0.4, nmsThreshold = 0.3, 
                    font_size=0.4, color_maps=None):

        rows, cols = frame.shape[:2]
        if self.counter_mode == 'area' :
            self.set_null_counter()
        self.frame_id += 1

        classIds = []
        confidences = []
        boxes = []

        for detection in np.array(outs)[0, 0, 0, :, :]:
            confidence = detection[2]
            classId = str(int(detection[1]))
            x = detection[3] * cols
            y = detection[4] * rows
            w = (detection[5] * cols) - x
            h = (detection[6] * rows) - y
            classIds.append(classId)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])

            # update counter
            if self.counter_mode == 'line' :
                self.counter_line_cross(classIds[i], x, y)
            else :
                self.counter_area(classIds[i])
            label_text = "%s (%.2f %%)" % (self.classes[classIds[i]], confidences[i])

            frame = utils.draw_ped(frame, label_text, x, y, x+w, y+h, 
                            font_size=font_size, 
                            color=color_maps[str(classIds[i])], 
                            text_color=(255,255,255))  
        return frame

    def draw_line(self, frame):
        if self.counter_mode == 'line' :
            frame = cv2.line(frame, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (255, 0, 255), 3)
        return frame