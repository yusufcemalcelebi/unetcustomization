from utilities import *
from model import *
from train import *

import numpy as np


data_path = './data'


def main():
    train(data_path)

    """
    count = 0
    for image, label in dataset_iterator:
        count += 1

    print(count)
    """
    # tools.rename_files()
    # tools.analyze_label_distribution()
    # tools.delete_images()
    # tools.find_dicom_range()


main()
