import mediapipe as mp
import numpy as np
import cv2 
import time
import pandas as pd


import os
#v2
class handPose():
    def __init__ (self, min_detection_confidence= .9, min_tracking_confidence=.9):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(min_detection_confidence=self.min_detection_confidence, 
                                        min_tracking_confidence = self. min_tracking_confidence)
        self.mpdraw = mp.solutions.drawing_utils

        self.bgModel = cv2.createBackgroundSubtractorMOG2(history= 0, varThreshold = 100, detectShadows=False)

    def cropHand(self, img, draw=False):
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img)
        img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hand_lms = results.multi_hand_landmarks
        x_min, x_max, y_min, y_max = (0,0,0,0)
        x_positions = []
        y_positions = []
        df = pd.DataFrame(columns = (range(42)))
        # print (hand_lms)
        if hand_lms :
            for lm in hand_lms[0].landmark:
                h, w, _ = img.shape
                cx, cy =  int(lm.x * w), int(lm.y * h)
                x_positions.append(cx)
                y_positions.append(cy)
            if len(x_positions) >=20:
                concatenation = np.c_[np.array([x_positions]),np.array([y_positions])]
                # print(concatenation)
                df = pd.DataFrame(concatenation, columns = (range(42))) 
                # print(df)
            # print(x_min, x_max, y_min, y_max)
            if draw:
                self.mpdraw.draw_landmarks(img, hand_lms[0], self.mphands.HAND_CONNECTIONS)
                
            # return 
        
        return img, df

def main():
    detector = handPose()
    cap = cv2.VideoCapture(0)
    while True:
        sucess, img =cap.read()
        img, dataframe = detector.cropHand(img, draw=True)
        print(dataframe)
        cv2.imshow("Image", img)
        if cv2.waitKey(10)==ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()