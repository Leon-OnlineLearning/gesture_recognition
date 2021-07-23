from EDA.HandPose import handPose
import os
import pandas as pd
import numpy as np
import cv2
data_path_v1 = '../real_data'
data_path_v2 = '/..real_datav2'
detector = handPose(.8, .8)
df = pd.DataFrame(columns = (range(42)))
df['target'] = ""

# v2
for i in range(1,3,1):
    path = f'../real_datav{i}'
    for j in ['/train/', '/test/', '/valid/']:
        path2 = path + j
        # print(path2)
        for n in os.listdir(path2):
            path3 = path2 + n
            print(path3)
            for l in os.listdir(path3):
                new_path = os.path.join(path3 , l)
                # print(new_path)
                img = cv2.imread(new_path)
                img, df2 = detector.cropHand(img, draw=True)
                if not df2.empty:
                    df2['target'] = n
                df = df.append(df2, ignore_index = True)
                # print(df)
path_to_save = '../file2.csv'
df.to_csv(path_to_save, index=False)