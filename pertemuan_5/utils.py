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
                    confThreshold = 0.4, nmsThreshold = 0.3, 
                    font_size=0.4, color_maps=None):

        rows, cols = frame.shape[:2]

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
                            color=color_maps[str(classIds[i])], 
                            text_color=(255,255,255))  

        return frame

    def postprocess_mask(self, frame, boxes, masks, classes, 
                        confThreshold = 0.4, maskThreshold = 0.3, 
                        font_size=0.4, color_maps=None, alpha=0.6):

        H, W = frame.shape[:2]

        for i in range(0, boxes.shape[2]):
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]

            if confidence > confThreshold:
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                x0, y0, x1, y1 = box.astype("int")
                boxW = x1 - x0
                boxH = y1 - y0

                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
                mask = mask > maskThreshold

                roi = frame[y0:y1, x0:x1][mask]
                color = color_maps[str(classID)]
                blended = (((1 - alpha) * np.array(color)) + (alpha * roi)).astype("uint8")
                frame[y0:y1, x0:x1][mask] = blended

                label_text = "%s (%.2f %%)" % (classes[str(classID)], (confidence*100))
                frame = self.draw_ped(frame, label_text, x0, y0, x1, y1, 
                            font_size=font_size, 
                            color=color_maps[str(classID)], 
                            text_color=(255,255,255)) 

                return frame