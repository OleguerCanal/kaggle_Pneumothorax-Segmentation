import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import pydicom
from mask_functions import mask2rle, rle2mask

class XRay:
    """ Class to hold an XRay information
    """
    def __init__(self, image_id, label, data_path):
        self.image_id = str(image_id)
        image_path = data_path + "/" + image_id + "/" + image_id + ".dcm"
        self.data = pydicom.dcmread(image_path)
        
        rows = int(self.data.Rows)
        cols = int(self.data.Columns)
        self.mask = rle2mask(label, rows, cols)
        pass

    def show_dcm_info(self):
        print("Filename.........:", file_path)
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
        plt.imshow(self.data.pixel_array, cmap = plt.cm.bone)
        plt.show()

    def plot_label(self, figsize = (10,10)):
        plt.figure(figsize = figsize)
        plt.imshow(self.mask, cmap = plt.cm.bone)
        plt.show()
    
    def plot_composition(self):
        #TODO(oleguer)
        pass

if __name__ == "__main__":
    labels_file = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/siim-acr-pneumothorax-segmentation-data/train-rle.csv"
    train_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/siim-acr-pneumothorax-segmentation-data/dicom-images-train/"

    labels_df = pd.read_csv(labels_file)

    for index, row in labels_df.iterrows():
        ImageID = row["ImageId"]
        EncodedPixels = str(row.values[1])  # Not sure why doesnt let me access by dict key row["EncodedPixels"]...
        xray = XRay(image_id = ImageID, label = EncodedPixels, data_path = train_path)
        xray.show_dcm_info()
        xray.plot_pixel_array()
        xray.plot_label()

        a = input() # To make the program stop