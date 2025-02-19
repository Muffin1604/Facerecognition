import numpy as np
import cv2
import os


#Face detection is 
def faceDetection(test_img, min_size=(210, 210), max_size=(380, 380)):#range 640 480
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar = cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml') 
    faces = face_haar.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=min_size,
        maxSize=max_size
    )
    return faces, gray_img
# def faceDetection(test_img):
#     gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
#     face_haar=cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml') 
#     faces=face_haar.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=3)
#     return faces,gray_img

#Labels for training data has been created

def labels_for_training_data(directory):
    faces=[]
    faceID=[]
    

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system file")
                continue
            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print ("img_path",img_path)
            print("id: ",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print ("Not Loaded Properly")
                continue

            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


#Here training Classifier is called
def train_classifier(faces,faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer


#Drawing a Rectangle on the Face Function
def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

#Putting text on images
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,255),2)
