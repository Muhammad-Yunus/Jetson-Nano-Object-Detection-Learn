import os
import cv2 
import numpy as np 

class Utils():
    def draw_ped(self, img, label, x0, y0, xt, yt, font_size=0.4, color=(255,127,0), text_color=(255,255,255)):

        y0, yt = max(y0 - 15, 0) , min(yt + 15, img.shape[0])
        x0, xt = max(x0 - 15, 0) , min(xt + 15, img.shape[1])

        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
        cv2.rectangle(img,
                        (x0, y0 + baseline),  
                        (max(xt, x0 + w), yt), 
                        color, 
                        2)
        cv2.rectangle(img,
                        (x0, y0 - h - baseline),  
                        (x0 + w, y0 + baseline), 
                        color, 
                        -1)
        cv2.rectangle(img,
                        (x0, y0 - h - baseline),  
                        (x0 + w, y0 + baseline), 
                        color, 
                        2)  
        cv2.putText(img, 
                    label, 
                    (x0, y0),                   
                    cv2.FONT_HERSHEY_SIMPLEX,     
                    font_size,                          
                    text_color,                
                    1,
                    cv2.LINE_AA) 
        return img

    def postprocess(self, outs, frame, classes, 
                    font_size=0.4, color=(255,127,0), text_color=(255,255,255), 
                    confThreshold = 0.4, nmsThreshold = 0.3):

        cols, rows = frame.shape[:2]

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

            label_text = "%s (%.2f %%)" % (classes[classIds[i]], confidences[i])
            frame = self.draw_ped(frame, label_text, x, y, x+w, y+h, 
                            font_size=font_size, 
                            color=color, text_color=text_color)  

        return frame