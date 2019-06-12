from utilities import *
from model import *
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend.tensorflow_backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_filepath = './log'


def train(data_path):
    # Check for the multiple path join
    train_image_path = os.path.join(data_path, 'train', 'images')
    train_label_path = os.path.join(data_path, 'train', 'labels')
    train_set_tools = utilities(train_image_path, train_label_path)

    train_image_iterator = train_set_tools.read_dicom_files()
    train_label_iterator = train_set_tools.read_nifti_files()
    train_iterator = zip(train_image_iterator, train_label_iterator)

    validation_image_path = os.path.join(data_path, 'validation', 'images')
    validation_label_path = os.path.join(data_path, 'validation', 'labels')
    validation_set_tools = utilities(
        validation_image_path, validation_label_path)

    val_image_iterator = validation_set_tools.read_dicom_files()
    val_label_iterator = validation_set_tools.read_nifti_files()
    validation_iterator = zip(val_image_iterator, val_label_iterator)

    model = unet()
    tb_cb = TensorBoard(log_dir=log_filepath)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        './model_v2.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit_generator(train_iterator,
                                  steps_per_epoch=600, epochs=5,
                                  validation_steps=10,
                                  validation_data=validation_iterator,
                                  callbacks=[model_checkpoint, tb_cb])
