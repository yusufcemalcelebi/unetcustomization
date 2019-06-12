import pydicom as pd
import nibabel as nib
import numpy as np
import os


class utilities:
    def __init__(self, image_path=None, label_path=None, img_rows=512, img_cols=512, num_classes=4, batch_size=4):
        self.image_path = image_path
        self.label_path = label_path
        self.img_rows = img_rows
        self.img_columns = img_cols
        self.num_classes = num_classes
        self.color_mode = 'grayscale'
        self.batch_size = batch_size

    def get_data_set_generator(self):
        train_image_generator = self.read_dicom_files()
        train_label_generator = self.read_nifti_files()

        train_generator = zip(train_image_generator, train_label_generator)

        return train_generator

    def read_dicom_files(self):
        patient_name_list = os.listdir(self.image_path)

        for patient_name in patient_name_list:
            patient_folder = os.path.join(self.image_path, patient_name)
            # print("\nReading Dicom for patient_id : " + patient_name)

            batch_index = 0
            for dicom_file in os.listdir(patient_folder):
                batch_index += 1
                dicom_file_path = os.path.join(patient_folder, dicom_file)

                ds = pd.dcmread(dicom_file_path)
                pixel_array = ds.pixel_array

                # resize for different shaped images
                if pixel_array.shape != (512, 512):
                    pixel_array = np.resize(
                        pixel_array, (self.img_rows, self.img_columns))

                normalized_image = self.get_normalized_image(
                    pixel_array, patient_name)

                if (batch_index == 1):
                    image_batch = np.zeros(
                        (0, self.img_rows, self.img_columns, 1))

                image_batch = np.append(image_batch, normalized_image)

                if (batch_index == self.batch_size):
                    yield image_batch.reshape(self.batch_size, self.img_rows, self.img_columns, 1)

                batch_index %= self.batch_size

    def read_nifti_files(self):
        file_name_list = os.listdir(self.label_path)

        for file_name in file_name_list:
            file_path = os.path.join(self.label_path, file_name)

            # print("\nReading Niftii for patient_id : " + file_name)

            nifti_file = nib.load(file_path)
            slices = nifti_file.get_fdata()

            slice_count = slices.shape[2]

            slices = slices.reshape(
                (slice_count, self.img_rows, self.img_columns))

            batch_index = 0
            for slice_index in range(slice_count):
                batch_index += 1
                slice = slices[slice_index, :, :]
                # resize for different shaped images
                if slice.shape != (512, 512):
                    slice = np.resize(
                        slice, (self.img_rows, self.img_columns))

                categorical_label = self.label_to_categorical(slice)

                if (batch_index == 1):
                    label_batch = np.zeros(
                        (0, self.img_rows, self.img_columns, self.num_classes))

                label_batch = np.append(label_batch, categorical_label)

                if (batch_index == self.batch_size):
                    yield label_batch.reshape(self.batch_size, self.img_rows, self.img_columns, self.num_classes)

                batch_index %= self.batch_size

    def analyze_label_distribution(self):
        label_iterator = self.read_nifti_files()
        counter = np.zeros(self.num_classes)

        for label_batch in label_iterator:
            for batch_index in range(self.batch_size):
                for i in range(self.img_rows):
                    for j in range(self.img_columns):
                        label_index = np.argmax(label_batch[batch_index, i, j])
                        counter[label_index] += 1

        total_pixel = np.sum(counter)

        for index, count in enumerate(counter):
            rate = count/total_pixel

            print("For label " + str(index) + " : %.5f" % rate)

    def rename_files(self):
        patient_folder_list = os.listdir(self.image_path)

        for patient_folder in patient_folder_list:
            patient_folder_path = os.path.join(self.image_path, patient_folder)

            for index, dicom_file_name in enumerate(os.listdir(patient_folder_path)):
                source_path = os.path.join(
                    patient_folder_path, dicom_file_name)
                destination_path = os.path.join(
                    patient_folder_path, str(index) + '-slice')

                os.rename(source_path, destination_path)

    # Delete images only contains background label
    def delete_images(self):
        file_name_list = os.listdir(self.label_path)
        slice_counts = {'1a.nii.gz': 42, '1c.nii.gz': 19, '2a.nii.gz': 30, '2c.nii.gz': 35,
                        '3a.nii.gz': 188, '4a.nii.gz': 42, '4c.nii.gz': 28, '5c.nii.gz': 46, '7b.nii.gz': 172}

        for file_name in file_name_list:
            file_path = os.path.join(self.label_path, file_name)
            patient_id = file_name.split('.')[0]

            nifti_file = nib.load(file_path)
            slices = nifti_file.get_fdata()
            slice_count = slices.shape[2]

            new_slices = np.ones((self.img_rows, self.img_columns,
                                  slice_counts[file_name]), dtype=np.int16)

            new_nifti_file_slice_index = 0
            for slice_index in range(slice_count):
                slice = slices[:, :, slice_index]
                uniqueValues, occurCount = np.unique(slice, return_counts=True)

                if (len(uniqueValues) > 1):
                    new_slices[:, :, new_nifti_file_slice_index] = slice
                    new_nifti_file_slice_index += 1
                else:
                    patient_folder = os.path.join(self.image_path, patient_id)
                    dicom_path = os.path.join(patient_folder, str(slice_index))

                    os.remove(dicom_path)

            new_nifti_file = nib.Nifti1Image(new_slices, np.eye(4))

            nib.save(new_nifti_file, os.path.join(
                './generated', file_name))

    def find_dicom_range(self):
        image_iterator = self.read_dicom_files()

        max_value = 0
        min_value = 9999999
        for image in image_iterator:

            local_min = np.amin(image)

            if(local_min < 0):
                local_max = np.amax(image)
                print(local_max)

                if(local_max > max_value):
                    max_value = local_max

            if(local_min < min_value):
                min_value = local_min

        print(max_value)
        print(min_value)

    def get_normalized_image(self, image, patient_id):
        image = image.astype(float)
        # 4a scans have different range
        if (patient_id == '4a'):
            min_value = -1500
            max_value = 1350

            image = (image - min_value) / (max_value - min_value)
        else:
            max_hounsfield_unit = 4095
            image /= max_hounsfield_unit

        return image

    def label_to_categorical(self, label):
        new_label = np.zeros(label.shape + (self.num_classes,))

        for i in range(self.num_classes):
            new_label[label == i, i] = 1

        return new_label
