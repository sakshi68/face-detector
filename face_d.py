#importing open cv
import cv2
#importing rand range from random for difficult color frame at screen
from random import randrange

# importing trained data for matching
trained_face_data=cv2.CascadeClassifier('face.xml')

#This can use for image also
#img=cv2.imread('p.jpg)

#using webcam as source where 0 is defaut source, Means our camera
webcam=cv2.VideoCapture(0)

#looping for each frame in the video
while True:
    s_F_R,frame=webcam.read()

    #for graying the image for better utilizing
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #taking coordinates
    cordinates=trained_face_data.detectMultiScale(frame)
    #loop on each face
    for(x,y,w,h) in cordinates:
        #accessing cordinates
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(255),randrange(254)),3)

    cv2.imshow('Face detector',frame)

    key=cv2.waitKey(1)

    if key==81 or key==113:
     break;
    
