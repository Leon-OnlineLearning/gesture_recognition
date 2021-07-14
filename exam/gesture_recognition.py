import numpy as np 
from loading_models import Singleton_model
import cv2
from tensorflow.keras.applications import mobilenet

def gesture_recognition(chunk_path):
    models = Singleton_model
    model, _ = models.getInstance
    predicted_classes = [] 

    vs = cv2.VideoCapture(str(chunk_path))
    read=0 #frame reading counter

    # loop over some frames
    while True:
        # grab the frame from the threaded video stream 
        (grabbed,frame) = vs.read()
        read += 1

        #When the video ends 
        if not grabbed:
            if predicted_classes.size != 0:
                gesture = max(set(predicted_classes), key = predicted_classes.count)
                return(gesture)
            return(-1)

        # check to see if we should process this frame
        if read % 10 == 0:
            #detection any hand in the frame
            hand = preprocessing(frame)
            if hand != 0:
                predicted_class = np.argmax(model.predict(hand),axis = 1) #return the predicted_class of the given hand
                predicted_classes.append(predicted_class)
            



def preprocessing(frame):
    models = Singleton_model
    _, detection = models.getInstance
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
        return ROI
    return 0