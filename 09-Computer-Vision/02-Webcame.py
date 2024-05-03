# capturing the webcam in gray scale
import numpy as np 
import cv2 as cv 

# cap object to capture video 
cap = cv.VideoCapture(0)
if cap.isOpened():
    ret,frame=cap.read()
else:
    ret = False
while ret:
    ret,frame= cap.read()
    
    # Convert frame to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    
    cv.imshow('Webcam (Grayscale)', gray_frame)
    if cv.waitKey(25) & 0xFF==ord("q"):
        break
cap.release()
cv.destroyAllWindows()

cv.destroyAllWindows()

