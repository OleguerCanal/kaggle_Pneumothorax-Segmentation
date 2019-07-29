from glob import glob
import numpy as np
import os
import random

from keras.preprocessing.image import ImageDataGenerator

import sys
from pathlib import Path
sys.path.append(str(Path(os.path.realpath(__file__)).parent.parent))
from data_processing.xray_reader import XRay

class DiskDataGenerator():
    """ Fetches augmented scans loaded from disk
    """
    def __init__(self, x_paths):
        self.xrays_paths = x_paths
        random.shuffle(self.xrays_paths)

        # y = [f for f in listdir(y_path) if isfile(join(y_path, f))]
        # self.zipped = zip(x, y)
        # self.zipped = random.shuffle(self.zipped)
        datagen_args = dict(
                        rotation_range = 20,
                        width_shift_range = 0.1,
                        height_shift_range = 0.1,
                        brightness_range = (0.75, 1.25),
                        shear_range = 0.1,
                        zoom_range = 0.1)
        self.datagen = ImageDataGenerator(**datagen_args)

        # Load an image and fit it
        xray = XRay(image_path = self.xrays_paths[0])
        x_train = np.array([xray.scan])
        self.im_height = xray.scan.shape[0]
        self.im_width = xray.scan.shape[1]
        x_train = x_train.reshape((-1, self.im_height, self.im_width, 1))
        self.datagen.fit(x_train)
   
    def __augment(self, x, y):
        self.datagen.flow(x, y, batch)
        return x, y

    def flow(self, batch_size):
        while True:
          # Select files (paths/indices) for the batch
          batch_paths = np.random.choice(a = self.xrays_paths, 
                                         size = batch_size)

          batch_input = []
          batch_output = []         
          # Read in each input, perform preprocessing and get labels
          for input_path in batch_paths:
              xray = XRay(image_path = input_path)
              batch_input.append(xray.scan)
              batch_output.append(xray.mask)

          # Return a tuple of (input,output) to feed the network
          batch_x = np.array(batch_input)
          batch_y = np.array(batch_output)

          batch_x = batch_x.reshape((-1, self.im_height, self.im_width, 1))
          batch_y = batch_y.reshape((-1, self.im_height, self.im_width, 1))

          for x_augmented, y_augmented in self.datagen.flow(batch_x, batch_y, batch_size):
            yield x_augmented, y_augmented
            break


if __name__ == "__main__":
    from matplotlib import pyplot

    data_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/input/train/images/1024/"
    train_paths = sorted(glob(data_path + "/dicom/*.png"))
    ddg = DiskDataGenerator(train_paths)
    for x, y in ddg.flow(2):
        print(x.shape)
        print(y.shape)
        print("##")
        pyplot.imshow(x[0, :, :, 0].reshape(1024, 1024), cmap=pyplot.get_cmap('gray'))
        pyplot.show()
        pyplot.imshow(y[0].reshape(1024, 1024), cmap=pyplot.get_cmap('gray'))
        pyplot.show()

