import pydicom as pd
import nibabel as nib
import numpy as np
import os


class utilities:
    def __init__(self, image_path=None, label_path=None, img_rows=512, img_cols=512, num_classes=4):
        self.image_path = image_path
        self.label_path = label_path
        self.img_rows = img_rows
        self.img_columns = img_cols
        self.num_classes = num_classes
        self.color_mode = 'gray'

    def read_dicom_files(self):
        file_name_list = os.listdir(self.image_path)

        for file_name in file_name_list:
            patient_folder = os.path.join(self.image_path, file_name)

            for dicom_file in os.listdir(patient_folder):
                dicom_file_path = os.path.join(patient_folder, dicom_file)

                ds = pd.dcmread(dicom_file_path)
                pixel_array = ds.pixel_array

                # resize for different shaped images
                if pixel_array.shape != (512, 512):
                    pixel_array = np.resize(
                        pixel_array, (self.img_rows, self.img_columns))

                yield pixel_array

    def read_nifti_files(self):
        file_name_list = os.listdir(self.label_path)

        for file_name in file_name_list:
            file_path = os.path.join(self.label_path, file_name)

            nifti_file = nib.load(file_path)
            slices = nifti_file.get_fdata()
            slice_count = slices.shape[2]

            for slice_index in range(slice_count):
                slice = slices[:, :, slice_index]

                yield slice

    def analyze_label_distribution(self):
        label_iterator = self.read_nifti_files()

        for label in label_iterator:
            counter = np.zeros(self.num_classes)

            uniqueValues, occurCount = np.unique(label, return_counts=True)

            for index, value in enumerate(uniqueValues):
                counter[int(value)] += occurCount[index]
            # print("Unique Values : " , uniqueValues)
            # print("Occurrence Count : ", occurCount)
        
        print(counter)
        # totalPixel = np.sum(counter)

        

