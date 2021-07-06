import os
import numpy as np
from os import listdir
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split





class flattenDataset():
    def __init__(self, img_size = 64, grayscale_image = True):
        self.img_size = img_size
        self.grayscale_image = grayscale_image

    def get_img(self, data_path):
        # reading the image
        img = cv2.imread(data_path)
        if self.grayscale_images:
            # print(data_path)
            img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def get_dataset(self, dataset_path, num_class, data_type = 'train'):
        # Getting all data from data path:
        labels = listdir(dataset_path) # Geting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = self.get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(i)
        # Create dateset:
        X = 1-np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists('./npy_dataset/'):
            os.makedirs('./npy_dataset/')
        np.save(f'./npy_dataset/X_{data_type}.npy', X)
        np.save(f'./npy_dataset/Y_{data_type}.npy', Y)


def main():
    flat_data = flattenDataset()
    img = flat_data.get_img('/img/path')

    train_path = r'../real_data/train'
    flat_data.get_dataset(train_path,10,'train')

    test_path = r'../real_data/test'
    flat_data.get_dataset(test_path,10,'train')

    valid_path = r'../real_data/valid'
    flat_data.get_dataset(valid_path,10,'train')

if __name__ == '__main__':
    main()