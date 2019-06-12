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

    for test_image, test_label in test_generator:
        results = model.predict(test_image)
