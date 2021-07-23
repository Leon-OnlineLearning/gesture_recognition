import mediapipe as mp
import numpy as np
import cv2 
import time


import os
#v1, v2
class handDetector():
    def __init__ (self, min_detection_confidence= .9, min_tracking_confidence=.9):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(min_detection_confidence=self.min_detection_confidence, 
                                        min_tracking_confidence = self. min_tracking_confidence)
        self.mpdraw = mp.solutions.drawing_utils

        self.bgModel = cv2.createBackgroundSubtractorMOG2(history= 0, varThreshold = 100, detectShadows=False)

    def findHands(self, img, draw=False):
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img)
        hand_lms = results.multi_hand_landmarks
        print (hand_lms)
        if hand_lms :
            if draw:
                self.mpdraw.draw_landmarks(img, hand_lms[0], self.mphands.HAND_CONNECTIONS)
        return img
    def cropHand(self, img, draw=False):
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img)
        img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hand_lms = results.multi_hand_landmarks
        x_min, x_max, y_min, y_max = (0,0,0,0)
        x_positions = []
        y_positions = []
        # print (hand_lms)
        if hand_lms :
            for lm in hand_lms[0].landmark:
                h, w, _ = img.shape
                cx, cy =  int(lm.x * w), int(lm.y * h)
                x_positions.append(cx)
                y_positions.append(cy)
            if len(x_positions) >=20:
                x = np.array(x_positions)
                y = np.array(y_positions)
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
            # print(x_min, x_max, y_min, y_max)
            if draw:
                self.mpdraw.draw_landmarks(img, hand_lms[0], self.mphands.HAND_CONNECTIONS)
                
            # return 
            
        return img, x_min, x_max, y_min, y_max 

def main():
    detector = handDetector()
    cap = cv2.VideoCapture(0)
    ptime = 0 
    while True:
        sucess, img =cap.read()
        img, x_min, x_max, y_min, y_max = detector.cropHand(img, draw=True)
        if x_min and  x_max and y_min and y_max:
            topLeft = (x_min - 35, y_min - 35)
            bottomRight = (x_max + 35,y_max + 35)
            cv2.rectangle(img, topLeft, bottomRight,(255,255,0),2)
        ctime = time.time()
        fps = 1 / (ctime-ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70),cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                    3,(255, 0, 255), 5)
        cv2.imshow("Image", img)
        if cv2.waitKey(10)==ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()