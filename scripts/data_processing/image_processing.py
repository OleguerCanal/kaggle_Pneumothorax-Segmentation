import cv2
import numpy as np

def process_mask(mask):
    # Dilation, according to this helps:
    # https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/99355#latest-572555
    dilation = cv2.dilate(img, 2, iterations = 1)
