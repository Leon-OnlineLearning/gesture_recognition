from EDA.HandDetector import handDetector
import cv2
import os
import time

# this script to collect real data _10 numbers with no white background"

detector = handDetector(.7, .7)
path  = r'testing/real_data'

for i in range(0,10):
    cap = cv2.VideoCapture(0)
    path2 = path+f'/{i}/'
    os.makedirs(path2, exist_ok=True)
    saved = 0
    while True:
        new_path = path2 + f'{saved}.png'
        sucess, img =cap.read()
        img, x_min, x_max, y_min, y_max = detector.cropHand(img, draw=False)
        cv2.putText(img,str(i) , (10, 70),cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                    3,(255, 0, 255), 5)
        cv2.imshow("Image", img)
        if x_min and  x_max and y_min and y_max:
            topLeft = (x_min - 10, y_min - 10)
            bottomRight = (x_max + 10,y_max + 10)
            x, y = topLeft[0], topLeft[1]
            w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
            ROI = img[y:y+h, x:x+w]
            if 0 not in ROI.shape:
                ROI = cv2.resize(ROI, (224,224))
                cv2.rectangle(img, topLeft, bottomRight,(255,255,0),2)
                cv2.imwrite(new_path, ROI)
                saved +=1
                print(saved)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or saved >=1000 :
            break
    
    time.sleep(4)
    cv2.destroyAllWindows()
    cap.release()
    print(saved)
