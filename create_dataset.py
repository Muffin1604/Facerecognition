import cv2
#import sys
cpt = 0
n=5

vidStream = cv2.VideoCapture(0)
while cpt <300:
    
    ret, frame = vidStream.read() # read frame and return code.
    if not ret:
        break
    #if cpt % n == 0:
    cv2.imshow("test window", frame)  # show image in window
    cv2.imwrite("train-images/2/image%04i.jpg" % cpt, frame)  # Save frame as image
    if cv2.waitKey(10) == ord('q'):
        break

    cpt += 1
    
    # cv2.imshow("test window", frame) # show image in window
    
    # cv2.imwrite("train-images/0/image%04i.jpg" %cpt, frame)    #Give path to  train-images/0/ and keep image%04i.jpg as it is in this line. Your images will be stored at train-images/0/ folder
    # cpt += 1

cv2.destroyAllWindows()
