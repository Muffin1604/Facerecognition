import numpy as np
import cv2
import os
import faceRecognition as fr
cap=cv2.VideoCapture(0)#If you want to recognise face from a video then replace 0 with video path
while True:
    ret,test_img=cap.read()
    if not ret:
        break
    #if c % n == 0:
    faces_detected,gray_img=fr.faceDetection(test_img)
    #print("face Detected: ",faces_detected)
    height, width = test_img.shape[:2]
    #print(height, width)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)
    cv2.imshow("face detection ", test_img)
    if cv2.waitKey(10)==ord('q'):
        break
cv2.destroyAllWindows()