from utilities import *

image_path = './images'
label_path = './labels'


def main():
    tools = utilities(image_path, label_path)
    """
    image_iterator = tools.read_dicom_files()
    label_iterator = tools.read_nifti_files()
    
    dataset_iterator = zip(image_iterator, label_iterator)

    for image, label in dataset_iterator:
        print(image.shape)
    """

    tools.analyze_label_distribution()


main()
