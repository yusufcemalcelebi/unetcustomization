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


def train(image_path, label_path):

    tools = utilities(image_path, label_path)

    image_iterator = tools.read_dicom_files()
    label_iterator = tools.read_nifti_files()

    dataset_iterator = zip(image_iterator, label_iterator)
    model = unet()
    tb_cb = TensorBoard(log_dir=log_filepath)

    # model_checkpoint = keras.callbacks.ModelCheckpoint(
    #    './model_v2.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit_generator(dataset_iterator,
                                  steps_per_epoch=200, epochs=30,
                                  callbacks=[tb_cb])
