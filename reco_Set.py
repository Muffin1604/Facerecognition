import numpy as np
import cv2
import os
import time
import pandas as pd
from datetime import datetime, timedelta

import faceRecognition as fr
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'trainingData.yml')  # Give path of where trainingData.yml is saved

cap = cv2.VideoCapture(0)  # If you want to recognize face from a video then replace 0 with video path

name = {0: "Karan", 1: "Ankush", 2: "Kalyan"}  # Change names accordingly.
columns = ['Timestamp', 'Name', 'ImagePath']
csv_file = 'recognized_faces.csv'
recognized_names = set()
frame_count = 0
last_unknown_time = datetime.now() - timedelta(minutes=2)

while True:
    ret, test_img1 = cap.read()
    test_img = cv2.flip(test_img1, 1)
    if not ret:
        break

    faces_detected, gray_img = fr.faceDetection(test_img)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)

    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + h]
        label, confidence = face_recognizer.predict(roi_gray)
        # print ("Confidence :",confidence)
        # print("label :",label)

        fr.draw_rect(test_img, face)
        if confidence < 35:
            predicted_name = name[label]
        else:
            predicted_name = "Unknown"

        fr.put_text(test_img, predicted_name, x, y)

        current_time = datetime.now()

        if predicted_name == "Unknown":
            if current_time - last_unknown_time >= timedelta(minutes=2):
                timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                frame_count += 1
                frame_file = f"frame-images/unknown/frame_{predicted_name}_{timestamp}.jpg"
                cv2.imwrite(frame_file, test_img)
                new_entry = pd.DataFrame({'Timestamp': [timestamp], 'Name': [predicted_name], 'ImagePath': [frame_file]})
                if not os.path.isfile(csv_file):
                    new_entry.to_csv(csv_file, index=False)
                else:
                    new_entry.to_csv(csv_file, mode='a', header=False, index=False)
                last_unknown_time = current_time
        elif predicted_name != "Unknown" and predicted_name not in recognized_names:
            timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
            frame_file = f"frame-images/recognised/frame_{predicted_name}_{timestamp}.jpg"
            cv2.imwrite(frame_file, test_img)
            new_entry = pd.DataFrame({'Timestamp': [timestamp], 'Name': [predicted_name], 'ImagePath': [frame_file]})
            if not os.path.isfile(csv_file):
                new_entry.to_csv(csv_file, index=False)
            else:
                new_entry.to_csv(csv_file, mode='a', header=False, index=False)
            recognized_names.add(predicted_name)

    cv2.imshow("face detection", test_img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()