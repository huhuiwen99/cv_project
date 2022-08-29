import os
import cv2
import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from network import Network

Batch_Size = 8
Image_Dim = (224, 224, 3)
Save_Interval = 50


def read_image(image_path):
    list_set = []
    image_num = 0
    # image_index = 0
    for folder_name in os.listdir(image_path):
        tmp_folder = image_path + '/' + folder_name
        for image_name in os.listdir(tmp_folder):
            img = cv2.imread(tmp_folder + '/' + image_name)
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
            img = np.reshape(img, Image_Dim)
            img = np.multiply(img, 1.0 / 255.0)
            list_set.append((img, int(folder_name[0])))
            image_num += 1
    return list_set, image_num

def plot_result(array, name):
    np.savetxt(name + '.txt', array)
    plt.figure()
    plt.title(name, fontsize = 16)
    plt.xlabel('episode', fontsize = 16)
    plt.ylabel(name, fontsize = 16)
    plt.plot(array)
    plt.savefig(name + '.png')
    plt.show()


def main():

    learning_rate = 0.01
    minimal_lr = 0.0001
    base_episode = 10

    train_list, train_num = read_image("TrainingSet")
    Label_Dim = 2

    agent = Network(input_dim = Image_Dim, output_dim = Label_Dim)

    average_loss = []
    average_accuracy = []

    for episode in range(500):
        print("episode: ", episode)

        if episode == base_episode - 1 and learning_rate >= minimal_lr:
            learning_rate = learning_rate / 10
            base_episode = base_episode * 10

        random.shuffle(train_list)
        all_loss = []
        all_accuracy = []
        for i in range(int(train_num / Batch_Size)):
            batch_image = [x for (x, y) in train_list[i * Batch_Size: (i+1) * Batch_Size]]
            batch_label = [y for (x, y) in train_list[i * Batch_Size: (i+1) * Batch_Size]]
            _, loss, accuracy = agent.sess.run([agent.optimize, agent.loss, agent.accuracy],
                feed_dict = {agent.input: batch_image, agent.label: batch_label, agent.lr: learning_rate})
            all_loss.append(loss)
            all_accuracy.append(accuracy)
            # print("step:{}, loss:{}, accuracy:{}".format(i, loss, accuracy))

        if train_num % Batch_Size != 0:
            batch_image = [x for (x, y) in train_list[-train_num % Batch_Size:]]
            batch_label = [y for (x, y) in train_list[-train_num % Batch_Size:]]
            _, loss, accuracy = agent.sess.run([agent.optimize, agent.loss, agent.accuracy],
                feed_dict = {agent.input: batch_image, agent.label: batch_label, agent.lr: learning_rate})
            all_loss.append(loss)
            all_accuracy.append(accuracy)
            # print("step:{}, loss:{}, accuracy:{}".format(int(train_num / Batch_Size), loss, accuracy))

        average_loss = np.mean(all_loss)
        average_accuracy = np.mean(all_accuracy)
        print("episode:{}, average_loss:{}, average_accuracy:{}".format(episode, average_loss, average_accuracy))

        if episode % Save_Interval == 0:
            agent.save_model(episode)

    plot_result(average_loss, "loss")
    plot_result(average_accuracy, "accuracy")

if __name__ == '__main__':
    main()