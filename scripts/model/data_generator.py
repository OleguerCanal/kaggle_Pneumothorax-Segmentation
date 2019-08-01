from glob import glob
import numpy as np
import os
import random
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator

import sys
from pathlib import Path
sys.path.append(str(Path(os.path.realpath(__file__)).parent.parent))
from data_processing.xray_reader import XRay

class DiskDataGenerator():
    """ Fetches augmented scans loaded from disk
    """
    def __init__(self, x_paths, batch_size):
        self.xrays_paths = x_paths[0:10000]
        self.batch_size = batch_size
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
   
    def get_val(self, val_ratio = 0.2):
        """ Returns np arrays of validation data (removing them from test)
        """
        print("Loading validation data")
        val_data = self.xrays_paths[0:int(val_ratio*len(self.xrays_paths))]
        self.xrays_paths = self.xrays_paths[int(val_ratio*len(self.xrays_paths)):]

        x_val = []
        y_val = []
        for path in tqdm(val_data):
            xray = XRay(image_path = path)
            x_val.append(xray.scan)
            y_val.append(xray.mask)


        x_val = np.array(x_val).reshape((-1, self.im_height, self.im_width, 1))
        y_val = np.array(y_val).reshape((-1, self.im_height, self.im_width, 1))
        return x_val, y_val


    def flow(self):
        while True:
          # Select files (paths/indices) for the batch
          batch_paths = np.random.choice(a = self.xrays_paths, 
                                         size = self.batch_size)

          batch_input = []
          batch_output = []         
          # Read in each input, perform preprocessing and get labels
          for input_path in batch_paths:
              xray = XRay(image_path = input_path)
              batch_input.append(xray.scan)
              batch_output.append(xray.mask)
              del xray

          # Return a tuple of (input,output) to feed the network
          batch_x = np.array(batch_input, dtype = np.float32)
          batch_y = np.array(batch_output, dtype = np.float32)

          batch_x = batch_x.reshape((-1, self.im_height, self.im_width, 1))
          batch_y = batch_y.reshape((-1, self.im_height, self.im_width, 1))
        #   print("yield")
          yield batch_x, batch_y

        #   for x_augmented, y_augmented in self.datagen.flow(batch_x, batch_y, batch_size):
        #     # print(x_augmented.shape)
        #     # print(y_augmented.shape)
        #     print("fetching")
        #     yield x_augmented, y_augmented
        #     del batch_x, batch_y
        #     break

    def get_steps(self):
        return (len(self.xrays_paths) // self.batch_size)


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

