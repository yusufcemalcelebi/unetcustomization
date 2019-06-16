from utilities import *
from train import *
# from test import *

import numpy as np


data_path = './data'


def main():
    train(data_path)
    """
    train_image_path = os.path.join(data_path, 'train', 'images')
    train_label_path = os.path.join(data_path, 'train', 'labels')
    train_set_tools = utilities(train_image_path, train_label_path)

    train_generator = train_set_tools.create_batch()

    for image, label in train_generator:
        pass
    """
    # tools.rename_files()
    # train_set_tools.analyze_label_distribution()
    # tools.delete_images()
    # tools.find_dicom_range()


main()
