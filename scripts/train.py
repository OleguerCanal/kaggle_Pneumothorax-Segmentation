import datetime
from glob import glob
import time
import os
from tqdm import tqdm

from data_processing.xray_reader import XRay
from model.architectures import simple_u_net
from model.metrics import dice_coef
from model.data_generator import DiskDataGenerator

import tensorflow as tf
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

# import keras.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4) #default is 1e-7

model_logging_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/models"
tensorboard_logging_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/logs"

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
    data_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/input/train/images/1024/"
    train_paths = sorted(glob(data_path + "/dicom/*.png"))

    # 1. Load architecture
    model = simple_u_net()

    # 3. Log model
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    save_path = str(model_logging_path) + "/" + str(time_stamp)
    log_model(path = save_path, model = model)

    # 4. Define model
    # Datagen
    datagen = DiskDataGenerator(train_paths)
    val_ratio = 0.2
    x_val, y_val = datagen.get_val(val_ratio = val_ratio)

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
    epochs = 100
    batch_size = 8
    history = model.fit_generator(
                        generator = datagen.flow(batch_size = batch_size),
                        epochs = epochs,
                        validation_data = (x_val, y_val),
                        verbose = 1,
                        callbacks = callbacks,
                        steps_per_epoch = (1 - val_ratio)*len(train_paths) // (batch_size))  # // is floor division