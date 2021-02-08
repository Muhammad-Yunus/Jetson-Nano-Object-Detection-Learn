import cv2 

cam = 0
cap = cv2.VideoCapture(cam)

while cap.isOpened():
    ret, img = cap.read()
    if not ret :
        break
    cv2.imshow("Stream", img)

    key = cv2.waitKey(10) 
    if key == ord('j') :
        cv2.imwrite("captured_photo.jpg", img)
    elif key == ord('q') :
        break

cv2.destroyAllWindows()
cap.release()