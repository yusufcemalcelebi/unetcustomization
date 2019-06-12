import os
from utilities import *
from model import *
import numpy as np
import cv2
import warnings


def test(data_path='./data', model_path='./model_v2.hdf5'):
    test_image_path = os.path.join(data_path, 'test', 'images')
    test_label_path = os.path.join(data_path, 'test', 'labels')

    test_set_tools = utilities(test_image_path, test_label_path)

    test_image_iterator = test_set_tools.read_dicom_files()

    model = load_model(model_path)

    for test_image in test_image_iterator:
        results = model.predict(test_image)
        predicted_image = results[0]
        print("\nNew image\n")

        for row in range(512):
            for column in range(512):
                if(np.argmax(predicted_image[row][column]) > 1):
                    print(np.argmax(predicted_image[row][column]))


test()
