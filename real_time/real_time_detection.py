import numpy as np 
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.models import  load_model
import cv2
from EDA.HandDetector import handDetector
import tensorflow as tf

#loading the gesture model
model = load_model('../Gesture recognition_Sign Language_mobilenet_')

#creating an object of hand detector class
detection = handDetector(.9, .9)

cap = cv2.VideoCapture(0)

while True:
    suc, frame = cap.read()
    img, x_min, x_max, y_min, y_max = detection.cropHand(frame)
    topLeft = (x_min - 35, y_min - 35)
    bottomRight = (x_max + 35,y_max + 35)
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

    #the location of the hand
    ROI = img[y:y+h, x:x+w]

    #to make sure that the hand is real
    if 0 not in ROI.shape:
        ROI = cv2.resize(ROI, (224,224))
        ROI = ROI.reshape(1,224,224,3) # return the image with shaping that TF wants.
        
        #preprocessing_the_image of the hand
        ROI = mobilenet.preprocess_input(ROI)

        predicted_class = np.argmax(model.predict(ROI),axis = 1) #return the predicted_class of the given hand
        
        cv2.rectangle(frame, topLeft, bottomRight,(255,255,0),2)
        cv2.putText(frame, str(predicted_class[0]), topLeft,cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                    3,(255, 0, 255), 5)
    cv2.imshow("frame", frame)
    if cv2.waitKey(10)==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()

