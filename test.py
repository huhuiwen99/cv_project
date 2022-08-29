from network import Network
import tensorflow as tf
import cv2
import numpy as np


def main():
    image = cv2.imread("TestSet/1-1.jpg", 0)
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)
    image = np.reshape(image, (1, 224, 224, 1))
    img_norm = np.zeros(image.shape, dtype = np.float)
    cv2.normalize(image, img_norm)

    agent = Network(input_dim = (224, 224, 1), output_dim = 2)
    agent.restore_model(50)
    output = agent.sess.run(agent.output, feed_dict = {agent.input: img_norm})

    print(output)

if __name__ == '__main__':
    main()