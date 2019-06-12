from utilities import *
from model import *
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_filepath = './log'


def train(data_path):
    # get train data
    train_image_path = os.path.join(data_path, 'train', 'images')
    train_label_path = os.path.join(data_path, 'train', 'labels')

    train_set_tools = utilities(train_image_path, train_label_path)
    train_iterator = train_set_tools.get_data_set_generator()

    # get validation data
    validation_image_path = os.path.join(data_path, 'validation', 'images')
    validation_label_path = os.path.join(data_path, 'validation', 'labels')

    validation_set_tools = utilities(
        validation_image_path, validation_label_path)
    validation_iterator = validation_set_tools.get_data_set_generator()

    model = unet()
    tb_cb = TensorBoard(log_dir=log_filepath)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        './model_v2.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit_generator(train_iterator,
                                  steps_per_epoch=8, epochs=100,
                                  validation_steps=8,
                                  validation_data=validation_iterator,
                                  callbacks=[model_checkpoint, tb_cb])
