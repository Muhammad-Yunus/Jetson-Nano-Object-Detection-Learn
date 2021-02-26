import cv2
import numpy as np

def crop_border(stitched):
    # create mask image from stitched image
    stitched = cv2.copyMakeBorder(stitched,  # create border
                                10, 10, 10, 10,
                                cv2.BORDER_CONSTANT, 
                                (0, 0, 0))
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)  # convert to gray
    __, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) # convert to binary image
    cnts, ___ = cv2.findContours(thresh.copy(), # find contour
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea) 
    (x, y, w, h) = cv2.boundingRect(c) 

    mask = np.zeros_like(thresh) 
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1) 

    # find minimum rectangle area with non-zero value
    minRect = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    cnts, __ = cv2.findContours(minRect.copy(), 
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    return stitched[y:y + h, x:x + w]