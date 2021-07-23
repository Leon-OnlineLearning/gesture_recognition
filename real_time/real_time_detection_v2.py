import cv2
from EDA.HandPose import handPose
from EDA.HandDetector import handDetector
import pickle

#loading the gesture model

data_path = '../finalized_model.sav'
 
# some time later...
 
# load the model from disk
knn_model = pickle.load(open(data_path, 'rb'))

#creating an object of hand detector class
pose_detection = handPose(.8, .8)
detector = handDetector(.8,.8)

cap = cv2.VideoCapture(0)

while True:
    suc, frame = cap.read()
    img, x_min, x_max, y_min, y_max = detector.cropHand(frame)
    topLeft = (x_min - 35, y_min - 35)
    bottomRight = (x_max + 35,y_max + 35)
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

    #the location of the hand
    ROI = img[y:y+h, x:x+w]

    #to make sure that the hand is real
    if 0 not in ROI.shape:
        ROI = cv2.resize(ROI, (224,224))
        img, df = pose_detection.cropHand(ROI, draw=True)
        # print(df) 
        if not df.empty:
        #preprocessing_the_image of the hand
            prediction = knn_model.predict(df)
            # print(prediction)

        
            cv2.rectangle(frame, topLeft, bottomRight,(255,255,0),2)
            cv2.putText(frame, str(prediction[0]), topLeft,cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                    3,(255, 0, 255), 5)
    cv2.imshow("frame", frame)
    if cv2.waitKey(10)==ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()

