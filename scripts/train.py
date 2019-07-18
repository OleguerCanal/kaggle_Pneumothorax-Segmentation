from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from data_processing.xray_reader import XRay
from data_processing.mask_functions import mask2rle  # Do I need to translate the image?
from model.architectures import simple_u_net
from model.metrics import dice_coef

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model_logging_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/models"
tensorboard_logging_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/logs"

def load_data(train_paths, train_val_prop = 0.2, rnd_seed = 1):
    x_train = []
    y_train = []

    for image_path in train_paths:
        xray = XRay(image_path = image_path)
        x_train.append(xray.scan)
        y_train.append(xray.mask)
        
        # Sizes TODO(oleguer): This shuold only be done once
        im_height = xray.scan.rows
        im_width = xray.scan.cols
        lab_height = xray.mask.rows
        lab_width = xray.mask.cols

    x_train = x_train.reshape((-1, im_height, im_width, 1))
    y_train = y_train.reshape((-1, lab_height, lab_width, 1))

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size = train_val_prop,
        random_state = rnd_seed)

    return x_train, x_val, y_train, y_val

def log_model(path):
    # Make sure dir exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Serialize model to JSON
    model_json = self.model.to_json()
    with open(path + "/architecture.json", "w") as json_file:
        json_file.write(model_json)

    # Save model params
    with open(path + "/params.yaml", 'w') as outfile:
        yaml.dump(self.params, outfile, default_flow_style=False)

if __name__ == "__main__":
    data_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/input/train/images/1024/"
    train_paths = sorted(glob(data_path + "/dicom/*.png"))

    # 0. Load data
    x_train, x_val, y_train, y_val = load_data(train_paths)

    # 1. Load architecture
    model = simple_u_net()

    # 3. Log model
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    save_path = str(model_logging_path) + "/" + str(time_stamp)
    log_model(path = save_path)

    # 4. Define model
    # Datagen
    datagen_args = dict(
                    rotation_range = 20,
                    width_shift_range = 0.1,
                    height_shift_range = 0.1,
                    brightness_range = (0.75, 1.25),
                    shear_range = 0.1,
                    zoom_range = 0.1)
    datagen = ImageDataGenerator(**datagen_args)
    datagen.fit(x_train)

    # Callbacks:
    weights_filepath = save_path + "/weights-{epoch:0f}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(  # Save model weights after each epoch
                                filepath=weights_filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')
    log_dir = str(tensorboard_logging_path) + "/{}".format(time.time())
    tensorboard = TensorBoard(log_dir = log_dir)
    learning_rate_reduction = ReduceLROnPlateau(
                                            monitor = 'val_acc', 
                                            patience = 5,
                                            verbose = 1,
                                            factor = 0.85,  # Each patience epoch reduce lr by half
                                            min_lr = 1e-10)
    callbacks = [checkpoint, learning_rate_reduction, tensorboard]

    # 4. Fit Model
    epochs = 100
    batch_size = 64
    history = self.model.fit_generator(
                        generator = datagen.flow(x_train, y_train, batch_size = batch_size),
                        epochs = epochs,
                        validation_data = (x_val, y_val),
                        verbose = 1,
                        callbacks = callbacks,
                        steps_per_epoch = x_train.shape[0] // batch_size)  # // is floor division