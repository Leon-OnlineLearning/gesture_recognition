import shutil
import os
import random

#to split our real data and the github data to train and test and valid data

RD_path = r'./testing/real_data/'

if os.path.isdir(RD_path+"train/0/") is False:
    os.makedirs(RD_path+'train',exist_ok=True)
    os.makedirs(RD_path+'valid', exist_ok=True)


for i in range(10):
    shutil.move(RD_path+f'{i}', RD_path+f'train/{i}')


    os.makedirs(RD_path+f'valid/{i}', exist_ok=True)
    os.makedirs(RD_path+f'test/{i}', exist_ok=True)

    
    valid_samples = random.sample(os.listdir(RD_path+f'train/{i}'), 50)
    for j in valid_samples:
        shutil.move(RD_path+f'train/{i}/{j}', RD_path+f'valid/{i}')
        
    
    test_samples = random.sample(os.listdir(RD_path+f'train/{i}'), 10)
    for k in test_samples:
        shutil.move(RD_path+f'train/{i}/{k}', RD_path+f'test/{i}')

