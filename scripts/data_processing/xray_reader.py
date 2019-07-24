import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import pydicom
import cv2
from glob import glob

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from mask_functions import mask2rle, rle2mask

class XRay:
    """ Class to hold an XRay information
    """
    def __init__(self, image_path):
        self.image_id = image_path.split("/")[-1].replace(".png", "")
        self.image_path = image_path
        self.label_path = image_path.replace("dicom", "mask")
        # self.data = pydicom.dcmread(self.image_path)
        # self.scan = self.data.pixel_array

        # rows = int(self.data.Rows)
        # cols = int(self.data.Columns)
        # self.mask = rle2mask(label, rows, cols)
        # self.mask
        # print(self.mask.shape)

        self.scan = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.mask = cv2.imread(self.label_path, cv2.IMREAD_GRAYSCALE)

    def show_dcm_info(self):
        # print("Filename.........:", self.image_path)
        # print("Storage type.....:", self.data.SOPClassUID)
        # pat_name = self.data.PatientName
        # display_name = pat_name.family_name + ", " + pat_name.given_name
        # print("Patient's name......:", display_name)
        # print("Patient id..........:", self.data.PatientID)
        # print("Patient's Age.......:", self.data.PatientAge)
        # print("Patient's Sex.......:", self.data.PatientSex)
        # print("Modality............:", self.data.Modality)
        # print("Body Part Examined..:", self.data.BodyPartExamined)
        # print("View Position.......:", self.data.ViewPosition)
        # if 'PixelData' in self.data:
        #     rows = int(self.data.Rows)
        #     cols = int(self.data.Columns)
        #     print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        #         rows=rows, cols=cols, size=len(self.data.PixelData)))
        #     if 'PixelSpacing' in self.data:
        #         print("Pixel spacing....:", self.data.PixelSpacing)
        # print()
        pass

    def plot_scan(self, figsize = (600, 600)):
        cv2.namedWindow('scan', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('scan', 600, 600)
        cv2.imshow('scan', self.scan)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_label(self, figsize = (10,10)):
        plt.figure(figsize = figsize)
        plt.imshow(self.mask, cmap = plt.cm.bone)
        plt.show()
    
    def plot_composition(self):
        rgb = cv2.cvtColor(self.scan, cv2.COLOR_GRAY2RGB)
        rgb = np.uint8(rgb)

        mask = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.uint8)
        mask[:,:, 2] = self.mask

        composition = cv2.addWeighted(rgb, 0.8, mask, 0.2, 0)

        cv2.namedWindow('composition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('composition', 600, 600)
        cv2.imshow('composition', composition)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# class XRayLabeller():
#     """Class to provide 
#     """
#     def __init__(self, labels_file):
#         self.labels_df = pd.read_csv(labels_file)

#     def get_id_label(self, image_path):
        # image_id = image_path.split("/")[-1].replace(".dcm", "")
        # label = self.labels_df[self.labels_df['ImageId'] == image_id]
        # label = label.values[0][1]  # Not sure why other stuff doesn work
        # return image_id, label

if __name__ == "__main__":
    data_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/input/train/images/1024/"

    # labeller = XRayLabeller(data_path + "train-rle.csv")
    train_paths = sorted(glob(data_path + "/dicom/*.png"))
    # test_paths = sorted(glob(data_path + "dicom-images-test/*/*/*.dcm"))
    
    for image_path in train_paths:
        xray = XRay(image_path = image_path)
        # xray.show_dcm_info()
        # xray.plot_label()
        xray.plot_composition()
        xray.plot_scan()
