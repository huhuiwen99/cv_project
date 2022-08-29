import cv2
import numpy as np
import os
from keras.models import load_model

class Pred(object):
    def __init__(self,filepath, names, image_size, modelfile):
        
        self.names = names
        self.filepath = filepath
        self.modelfile = modelfile
        self.image_size = image_size
        self.images = []
        self.image_name = []

    def read_data(self):
        for each in os.listdir(self.filepath):
            image = cv2.imread(self.filepath + '/' + each)
            image = cv2.resize(image, (self.image_size,self.image_size), interpolation = cv2.INTER_CUBIC)
            image = np.array(image).reshape(-1,self.image_size,self.image_size,3).astype("float32")/255
            self.images.append(image)
            self.image_name.append(each)


    def pred_data(self):
        model = load_model(self.modelfile)
        for img, img_name in zip(self.images, self.image_name):
            print(img_name)
            prediction = model.predict(img)
            count = 0
            for i in prediction[0]:
                percent = '%.5f%%'%(i*100)
                print(f"{self.names[count]}的概率：{percent}")
                count += 1



pred = Pred(filepath = "TestSet", names = ["烈性犬","温和犬"], image_size = 64, modelfile="my_model/my_model.h6")
pred.read_data()
pred.pred_data()