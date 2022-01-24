import cv2
import numpy as np
from utils import Utils

utils = Utils()

class Counter():
    def __init__(self, classes, mode='area', lines=[], color_lines=[], threshDist = 50, direction=('left', 'right')):
        self.classes = classes
        self.frame_id = 0
        self.counter_objects = []
        self.counter_mode = mode
        self.lines = lines 
        self.threshDist = threshDist
        self.direction = direction
        self.set_null_counter()
        self.color_lines = color_lines

    def set_null_counter(self):
        self.counter_objects = []
        for __ in self.lines :
            class_counter = {}
            for class_id in self.classes:
                class_counter[class_id] = {
                                            'in' : False,
                                            'frame_id' : 0,
                                            'counter' : 0     
                                            }  
            self.counter_objects.append(class_counter) 


    def shortest_distance(self, x, y, line):  
        x1, y1, x2, y2 = line
        dx = x2-x1
        dy = y2-y1
        m = np.sqrt(np.square(dx) + np.square(dy))
        r = abs(dx*(y1-y) - (x1-x)*dy) / m 
        return r

    def counter_area(self, class_id):
        if self.counter_objects[0][class_id]['frame_id'] == self.frame_id :
            self.counter_objects[0][class_id]['counter'] += 1
        else :
            self.counter_objects[0][class_id]['counter'] = 1
            self.counter_objects[0][class_id]['frame_id'] = self.frame_id

    def counter_line_cross(self, class_id, x, y):
        r = self.shortest_distance(x, y, self.lines[0])
        if r < self.threshDist :
            if self.counter_objects[0][class_id]['in'] == False :
                self.counter_objects[0][class_id]['counter'] += 1
            self.counter_objects[0][class_id]['in'] = True 
        else :
            self.counter_objects[0][class_id]['in'] = False

    def counter_multiline_cross(self, class_id, x, y):
        for i, line in enumerate(self.lines) :
            r = self.shortest_distance(x, y, line)
            if r < self.threshDist :
                if self.counter_objects[i][class_id]['in'] == False :
                    self.counter_objects[i][class_id]['counter'] += 1
                self.counter_objects[i][class_id]['in'] = True 
            else :
                self.counter_objects[i][class_id]['in'] = False

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

            cx = x + int(w/2)
            cy = y + int(h/2)

            # update counter
            if self.counter_mode == 'line' :
                self.counter_line_cross(classIds[i], cx, cy)
            elif self.counter_mode == 'multiline' :
                self.counter_multiline_cross(classIds[i], cx, cy)
            else :
                self.counter_area(classIds[i])
            label_text = "%s (%.2f%%)" % (self.classes[classIds[i]], confidences[i])

            frame = utils.draw_ped(frame, label_text, x, y, x+w, y+h, 
                            font_size=font_size, 
                            color=color_maps[str(classIds[i])], 
                            text_color=(255,255,255))  
        return frame

    def draw_line(self, frame):
        if self.counter_mode == 'line' :
            x1, y1, x2, y2 = self.lines[0]
            frame = cv2.line(frame, (x1, y1), (x2, y2 ), self.color_lines[0], 3)
        elif self.counter_mode == 'multiline':
            for i, line in enumerate(self.lines) :
                x1, y1, x2, y2 = line
                frame = cv2.line(frame, (x1, y1), (x2, y2 ), self.color_lines[i], 3)          
        return frame