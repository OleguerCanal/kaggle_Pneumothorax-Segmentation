import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import pydicom
from mask_functions import mask2rle, rle2mask
import cv2

class XRay:
    """ Class to hold an XRay information
    """
    def __init__(self, image_id, label, data_path):
        self.image_id = str(image_id)
        image_path = data_path + "/" + image_id + "/" + image_id + ".dcm"
        self.image_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/siim-acr-pneumothorax-segmentation-data/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.300.1517875162.258080/1.2.276.0.7230010.3.1.3.8323329.300.1517875162.258079/1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081.dcm"
        self.data = pydicom.dcmread(self.image_path)
        self.scan = self.data.pixel_array

        print(label == " -1")
        print(type(label))
        rows = int(self.data.Rows)
        cols = int(self.data.Columns)
        self.mask = rle2mask(label, rows, cols)
        print(self.mask.shape)
        

    def show_dcm_info(self):
        print("Filename.........:", self.image_path)
        print("Storage type.....:", self.data.SOPClassUID)
        print()

        pat_name = self.data.PatientName
        display_name = pat_name.family_name + ", " + pat_name.given_name
        print("Patient's name......:", display_name)
        print("Patient id..........:", self.data.PatientID)
        print("Patient's Age.......:", self.data.PatientAge)
        print("Patient's Sex.......:", self.data.PatientSex)
        print("Modality............:", self.data.Modality)
        print("Body Part Examined..:", self.data.BodyPartExamined)
        print("View Position.......:", self.data.ViewPosition)
        
        if 'PixelData' in self.data:
            rows = int(self.data.Rows)
            cols = int(self.data.Columns)
            print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
                rows=rows, cols=cols, size=len(self.data.PixelData)))
            if 'PixelSpacing' in self.data:
                print("Pixel spacing....:", self.data.PixelSpacing)

    def plot_scan(self, figsize = (10,10)):
        plt.figure(figsize = figsize)
        plt.imshow(self.scan, cmap = plt.cm.bone)
        plt.show()

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

if __name__ == "__main__":
    labels_file = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/siim-acr-pneumothorax-segmentation-data/train-rle.csv"
    train_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/siim-acr-pneumothorax-segmentation-data/dicom-images-train/"

    labels_df = pd.read_csv(labels_file)

    for index, row in labels_df.iterrows():
        ImageID = row["ImageId"]
        EncodedPixels = str(row.values[1])  # Not sure why doesnt let me access by dict key row["EncodedPixels"]...
        xray = XRay(image_id = ImageID, label = EncodedPixels, data_path = train_path)
        xray.show_dcm_info()
        # xray.plot_scan()
        # xray.plot_label()
        xray.plot_composition()