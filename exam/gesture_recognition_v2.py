from exam.loading_models_v2 import Singleton_model
import cv2

def gesture_recognition_V2(chunk_path):
    # models = Singleton_model()
    # print(models.getInstance.Singleton_model.__model, models.getInstance.Singleton_model.__detector)
    model, _, pose_detection = Singleton_model.getInstance()
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
            if len(predicted_classes):
                gesture = max(set(predicted_classes), key = predicted_classes.count)
                return gesture
            return(-1)

        # check to see if we should process this frame
        if read % 10 == 0:
            # print(read)
            #detection any hand in the frame
            hand = preprocessing(frame)
            if (type(hand) != int):
                img, df = pose_detection.cropHand(hand, draw=True)
                # print(df) 
                if not df.empty:
                #preprocessing_the_image of the hand
                    prediction = model.predict(df)
                    predicted_classes.append(prediction[0].item())
            



def preprocessing(frame):
    # models = Singleton_model()
    _, detection,_ = Singleton_model.getInstance()
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
        return ROI
    return 0