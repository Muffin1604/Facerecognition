import numpy as np
import cv2
import os

import faceRecognition as fr
#print (fr)
faces,faceID=fr.labels_for_training_data(r'train-images') #Give path to the train-images folder which has both labeled folder as 0 and 1
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'trainingData.yml') #It will save the trained model. Just give path to where you want to save
print("The model is trained successfully")
