import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pydicom

class Xray:
    """ Class to hold an XRay information
    """
    def __init__(self, image_path, csv_path):
        self.
        self.data = pydicom.dcmread(image_path)
        # self.mask = 
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

    def plot_pixel_array(self, figsize=(10,10)):
        plt.figure(figsize=figsize)
        plt.imshow(self.data.pixel_array, cmap=plt.cm.bone)
        plt.show()
