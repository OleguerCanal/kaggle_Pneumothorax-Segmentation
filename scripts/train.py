import datetime
import numpy as np
from glob import glob
import time
import os
from tqdm import tqdm
from PIL import Image

from data_processing.xray_reader import XRay
from model.architectures import simple_u_net
from model.metrics import dice_coef
from model.data_generator import DiskDataGenerator

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

import tensorflow as tf
print(tf.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# from keras import backend as K
# cfg = K.tf.ConfigProto()
# cfg.gpu_options.allow_growth = True
# K.set_session(K.tf.Session(config=cfg))

# import keras.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4) #default is 1e-7

model_logging_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/models"
tensorboard_logging_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/logs"

def load_data(train_paths, train_val_prop = 0.2, rnd_seed = 1):
    x_train = []
    y_train = []

    train_paths = train_paths[0:10]
    for image_path in tqdm(train_paths):
        xray = XRay(image_path = image_path)
        # xray.plot_composition()
        x_train.append(xray.scan)
        y_train.append(xray.mask)
        del xray

    im_height = x_train[0].shape[0]
    im_width = x_train[0].shape[1]
    lab_height = y_train[0].shape[0]
    lab_width = y_train[0].shape[1]

    x_train = np.array(x_train).reshape((-1, im_height, im_width, 1))
    y_train = np.array(y_train).reshape((-1, lab_height, lab_width, 1))

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size = train_val_prop,
        random_state = rnd_seed)

    return x_train, x_val, y_train, y_val

def log_model(path, model):
    # Make sure dir exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Serialize model to JSON
    model_json = model.to_json()
    with open(path + "/architecture.json", "w") as json_file:
        json_file.write(model_json)

    # Save model params
    # with open(path + "/params.yaml", 'w') as outfile:
    #     yaml.dump(self.params, outfile, default_flow_style=False)

if __name__ == "__main__":
    # data_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/input/train/images/64/"
    data_path = "/media/oleguer/Extenció/FEINA/projectes/pneumotorax/input/train/images/256/"
    train_paths = sorted(glob(data_path + "/dicom/*.png"))


    # 1. Load architecture
    model = simple_u_net()

    # 3. Log model
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    save_path = str(model_logging_path) + "/" + str(time_stamp)
    log_model(path = save_path, model = model)

    # 4. Define model

    # Datagen
    val_ratio = 5e-4
    batch_size = 2
    datagen = DiskDataGenerator(train_paths, batch_size = batch_size)
    x_val, y_val = datagen.get_val(val_ratio = val_ratio)
    # x_train, x_val, y_train, y_val = load_data(train_paths)
    # datagen_args = dict(
    #             rotation_range = 5,
    #             width_shift_range = 0.1,
    #             height_shift_range = 0.1,
    #             brightness_range = (0.9, 1.1),
    #             shear_range = 0.1,
    #             zoom_range = 0.1)
    # datagen = ImageDataGenerator(**datagen_args)
    # datagen.fit(x_train)



    # Callbacks:
    weights_filepath = save_path + "/weights-{epoch:0f}-{dice_coef:.4f}.hdf5"
    checkpoint = ModelCheckpoint(  # Save model weights after each epoch
                                filepath=weights_filepath,
                                monitor='dice_coef',
                                verbose=1,
                                save_best_only=True,
                                mode='max')
    log_dir = str(tensorboard_logging_path) + "/{}".format(time.time())
    tensorboard = TensorBoard(log_dir = log_dir)
    learning_rate_reduction = ReduceLROnPlateau(
                                            monitor = 'dice_coef', 
                                            patience = 5,
                                            verbose = 1,
                                            factor = 0.85,  # Each patience epoch reduce lr by half
                                            min_lr = 1e-10)
    callbacks = [checkpoint, learning_rate_reduction, tensorboard]

    # 4. Fit Model
    epochs = 200
    history = model.fit_generator(
                        generator = datagen.flow(),
                        epochs = epochs,
                        validation_data = (x_val, y_val),
                        verbose = 1,
                        callbacks = callbacks,
                        # steps_per_epoch = (1 - val_ratio)*len(train_paths) // (batch_size))  # // is floor division
                        steps_per_epoch = datagen.get_steps())  # // is floor division

    # history = model.fit_generator(
    #                         generator = datagen.flow(x_train, y_train, batch_size = batch_size),
    #                         epochs = epochs,
    #                         validation_data = (x_val, y_val),
    #                         verbose = 1,
    #                         callbacks = callbacks,
    #                         steps_per_epoch = x_train.shape[0] // batch_size)  # // is floor division