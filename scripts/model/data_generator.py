from data_processing.xray_reader import XRay
from glob import glob
import numpy as np
import os
import random
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator

import sys
from pathlib import Path
sys.path.append(str(Path(os.path.realpath(__file__)).parent.parent))


class DiskDataGenerator():
    """ Fetches augmented scans loaded from disk
    """

    def __init__(self, x_paths, val_ratio=0.2, train_batch_size=1, val_batch_size=1):
        self.xrays_paths = x_paths
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        random.shuffle(self.xrays_paths)

        # y = [f for f in listdir(y_path) if isfile(join(y_path, f))]
        # self.zipped = zip(x, y)
        # self.zipped = random.shuffle(self.zipped)
        datagen_args = dict(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=(0.8, 1.2),
            shear_range=0.1,
            zoom_range=0.1)
        self.datagen = ImageDataGenerator(**datagen_args)

        # Load an image and fit it
        xray = XRay(image_path=self.xrays_paths[0])
        x_train = np.array([xray.scan])
        self.im_height = xray.scan.shape[0]
        self.im_width = xray.scan.shape[1]
        x_train = x_train.reshape((-1, self.im_height, self.im_width, 1))
        self.datagen.fit(x_train)

        # Split train/val
        self.val_paths = self.xrays_paths[0:int(
            val_ratio*len(self.xrays_paths))]
        self.train_paths = self.xrays_paths[int(
            val_ratio*len(self.xrays_paths)):]
        del self.xrays_paths

    def flow_train(self):
        while True:
            # Select files (paths/indices) for the batch
            batch_paths = np.random.choice(a=self.train_paths,
                                           size=self.train_batch_size)

            batch_input = []
            batch_output = []
            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                xray = XRay(image_path=input_path)
                batch_input.append(xray.scan)
                batch_output.append(xray.mask)
                del xray

            # Return a tuple of (input,output) to feed the network
            batch_x = np.array(batch_input, dtype=np.float32)
            batch_y = np.array(batch_output, dtype=np.float32)

            batch_x = batch_x.reshape((-1, self.im_height, self.im_width, 1))
            batch_y = batch_y.reshape((-1, self.im_height, self.im_width, 1))
        #   print("yield")
        #   yield batch_x, batch_y

            for x_augmented, y_augmented in self.datagen.flow(batch_x, batch_y, self.train_batch_size):
                # print(x_augmented.shape)
                # print(y_augmented.shape)
                # print("fetching")
                yield x_augmented, y_augmented
                del batch_x, batch_y
                break

    def flow_val(self):
        while True:
            # Select files (paths/indices) for the batch
            batch_paths = np.random.choice(a=self.val_paths,
                                           size=self.val_batch_size)

            batch_input = []
            batch_output = []
            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                xray = XRay(image_path=input_path)
                batch_input.append(xray.scan)
                batch_output.append(xray.mask)
                del xray

            # Return a tuple of (input,output) to feed the network
            batch_x = np.array(batch_input, dtype=np.float32)
            batch_y = np.array(batch_output, dtype=np.float32)

            batch_x = batch_x.reshape((-1, self.im_height, self.im_width, 1))
            batch_y = batch_y.reshape((-1, self.im_height, self.im_width, 1))
        #   print("yield")
        #   yield batch_x, batch_y

            for x_augmented, y_augmented in self.datagen.flow(batch_x, batch_y, self.val_batch_size):
                # print(x_augmented.shape)
                # print(y_augmented.shape)
                # print("fetching")
                yield x_augmented, y_augmented
                del batch_x, batch_y
                break

    def get_train_steps(self):
        return (len(self.train_paths) // self.train_batch_size)

    def get_val_steps(self):
        return (len(self.val_paths) // self.val_batch_size)


if __name__ == "__main__":
    from matplotlib import pyplot

    data_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/input/train/images/1024/"
    train_paths = sorted(glob(data_path + "/dicom/*.png"))
    ddg = DiskDataGenerator(train_paths)
    for x, y in ddg.flow(2):
        print(x.shape)
        print(y.shape)
        print("##")
        # pyplot.imshow(x[0, :, :, 0].reshape(1024, 1024), cmap=pyplot.get_cmap('gray'))
        # pyplot.show()
        # pyplot.imshow(y[0].reshape(1024, 1024), cmap=pyplot.get_cmap('gray'))
        # pyplot.show()
