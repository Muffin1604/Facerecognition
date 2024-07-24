import cv2
#import sys
cpt = 0
n=20

vidStream = cv2.VideoCapture(0)
while True:
    
    ret, frame1 = vidStream.read() # read frame and return code.
    frame = cv2.flip(frame1,1)
    cv2.imshow("test window", frame)  # show image in window
    if not ret:
        break
    if cv2.waitKey(5) == ord('k'):
        cv2.imwrite("train-images/2/image%04i.jpg" % cpt, frame)  # Save frame as image
        cpt += 1
    if cv2.waitKey(10) == ord('q'):
        break

    if cpt == 30:
        break
    
    # cv2.imshow("test window", frame) # show image in window
    
    # cv2.imwrite("train-images/0/image%04i.jpg" %cpt, frame)    #Give path to  train-images/0/ and keep image%04i.jpg as it is in this line. Your images will be stored at train-images/0/ folder
    # cpt += 1
print("Dataset created successfully")
cv2.destroyAllWindows()
