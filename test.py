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
        """
        print("\nLabel distribution for predicted :")
        label_distribution(results)

        print("\Label distribution for real")
        label_distribution(test_label)
        """
        calculate_dice_metric(test_label, results)
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


def calculate_dice_metric(real_label_batch, predicted_label_batch):
    true_prediction_counts = np.zeros((4))
    real_label_counts = np.zeros((4))
    total_pixel = 512*512

    for batch_index in range(4):
        real_label = real_label_batch[batch_index]
        predicted_label = predicted_label_batch[batch_index]

        for i in range(512):
            for j in range(512):
                real = np.argmax(real_label[i, j])
                predict = np.argmax(predicted_label[i, j])

                real_label_counts[real] += 1
                if (real == predict):
                    true_prediction_counts[real] += 1

    for i in range(4):
        print(true_prediction_counts[i] / real_label_counts[i])
    print("\n")


test()
