import numpy as np
import cv2
import os
import faceRecognition as fr

# Define paths
model_path = 'trainingData.yml'
new_data_path = 'train-images'  # Path to new user's dataset

# Load existing model if available
if os.path.exists(model_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)
    print("Existing model loaded successfully")
else:
    # If no model exists, create a new one
    faces, faceID = fr.labels_for_training_data(new_data_path)
    face_recognizer = fr.train_classifier(faces, faceID)
    face_recognizer.save(model_path)
    print("New model trained and saved successfully")
    exit()

# Get new user's data
new_faces, new_faceID = fr.labels_for_training_data(new_data_path)

# Update the model with new data
face_recognizer.update(new_faces, np.array(new_faceID))

# Save the updated model
face_recognizer.save(model_path)
print("Model updated with new user data and saved successfully")



# import numpy as np
# import cv2
# import os

# import faceRecognition as fr
# #print (fr)
# faces,faceID=fr.labels_for_training_data(r'train-images') #Give path to the train-images folder which has both labeled folder as 0 and 1
# face_recognizer=fr.train_classifier(faces,faceID)
# face_recognizer.save(r'trainingData.yml') #It will save the trained model. Just give path to where you want to save
# print("The model is trained successfully")
