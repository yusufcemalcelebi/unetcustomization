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

    test_image_iterator = test_set_tools.read_dicom_files()
    test_label_iterator = test_set_tools.read_nifti_files()

    test_iterator = zip(test_image_iterator, test_label_iterator)

    model = load_model(model_path)
    counter = 0
    for test_image, test_label in test_iterator:
        print("Next image")
        results = model.predict(test_image)

        predicted_mask = results[0]

        for i in range(512):
            for j in range(512):
                label = np.argmax(predicted_mask[i, j])
                if(label != 0):
                    print(label)

        counter += 1
        if (counter == 3):
            break


test()
