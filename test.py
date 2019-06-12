import os
from utilities import *
from model import *
import numpy as np
import cv2
import warnings
from sklearn.metrics import classification_report


def test(data_path='./data', model_path='./model_v2.hdf5'):
    test_image_path = os.path.join(data_path, 'test', 'images')
    test_label_path = os.path.join(data_path, 'test', 'labels')

    test_set_tools = utilities(test_image_path, test_label_path)
    test_generator = test_set_tools.get_data_set_generator()

    model = load_model(model_path)

    count = 0
    for test_image, test_label in test_generator:
        count += 1
        results = model.predict(test_image)

        label_distribution(results)
        if count == 7:
            break


def label_distribution(label_batch):
    counter = np.zeros((4))
    for label in label_batch:
        for i in range(512):
            for j in range(512):
                label_index = np.argmax(label[i, j])

                counter[label_index] += 1
    print(counter)


test()
