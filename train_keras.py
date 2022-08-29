import numpy as np
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Activation,Conv2D,MaxPool2D,Flatten,Dense,Dropout
from keras.models import Sequential
import os
import cv2
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

def read_image(image_path):
    list_set = []
    image_num = 0
    image_index = []
    for folder_name in os.listdir(image_path):
        tmp_folder = image_path + '/' + folder_name
        # print(folder_name)
        for image_name in os.listdir(tmp_folder):
            # print(image_name)
            img = cv2.imread(tmp_folder + '/' + image_name)
            img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_CUBIC)
            img = np.reshape(img, (64, 64, 3))
            img = np.multiply(img, 1.0 / 255.0)
            list_set.append(img)
            init_array = np.zeros(2)
            init_array[int(folder_name[0])] = 1.0
            image_index.append(init_array)
            image_num += 1
    list_set = np.array(list_set)
    image_index = np.array(image_index)
    return list_set, image_index, image_num


class Train():
    def __init__(self, epoch, batch_size):
        
        self.epoch = epoch
        self.batch_size = batch_size

    def read_data(self, train_path, val_path):
        self.X_train, self.Y_train, train_num = read_image(train_path)
        self.X_val, self.Y_val, val_num = read_image(val_path)

    def train_data(self):
        model = Sequential([Conv2D(filters=16,kernel_size=(5,5),padding='same',activation='relu',input_shape=(64,64,3)),
                            MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'),
                            Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu'),
                            MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'),
                            Dropout(0.5),
                            Flatten(),
                            Dense(64,activation='relu'),
                            Dropout(0.25),
                            Dense(2,activation='softmax'),])
        model.compile(loss = categorical_crossentropy, optimizer = Adadelta(), metrics = ['accuracy'])
        self.history = model.fit(self.X_train, self.Y_train, batch_size = self.batch_size, epochs = self.epoch,
                                    validation_data = (self.X_val, self.Y_val))
        model.save("my_model/my_model.h6")


agent = Train(22, 16)
agent.read_data(train_path = "TrainingSet", val_path = "ValidationSet")
agent.train_data()
