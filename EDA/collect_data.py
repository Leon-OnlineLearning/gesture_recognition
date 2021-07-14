import shutil
import os

RD_path = r'./testing/real_data/'
dataset_path = r'./testing/dataset/'

# to collect the keggle dataset with our real dataset which we've collected to make our model more robust 


for i in range(10):

    for j in (os.listdir(dataset_path+f'train/{i}')):
        shutil.move(dataset_path+f'train/{i}/{j}', RD_path+f'{i}')

    for j in (os.listdir(dataset_path+f'train/{i}')):
        shutil.move(dataset_path+f'test/{i}/{j}', RD_path+f'{i}')

    for j in (os.listdir(dataset_path+f'train/{i}')):
        shutil.move(dataset_path+f'valid/{i}/{j}', RD_path+f'{i}')
    
    