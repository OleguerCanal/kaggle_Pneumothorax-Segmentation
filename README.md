# kaggle_Pneumothorax-Segmentation
My solution (attempt) for the SIIM-ACR Pneumothorax Segmentation kaggle: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview
**IMPORTANT NOTE:** This is more an exercise for me to learn how to set up a DL environment, project and test some 

## Download data
To download the raw data, either run system_setup/install_google_dependencies.sh and then copy kaggle_provided_data/data_processing/download_images.py to raw_data folder and run it.

Or just download and extract from here:
https://www.kaggle.com/jesperdramsch/siim-acr-pneumothorax-segmentation-data/downloads/siim-acr-pneumothorax-segmentation-data.zip/1

## Environment
Set -up your machine with nvidia drivers, CUDA, cuDNN:
https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/


`mkvirtualenv -p python3 pneumo`

`workon pneumo`

`pip install -r keras_tf_requirements.txt`

`pip install -r custom_requirements.txt`

## Tested Configuration:
GPU: nvidia RTX 2070
Nvidia Drivers: 430.40
CUDA: 10.0
cuDNN: 7.6.2.24 for CUDA 10.0
Tensorflow: 1.14.0 (GPU)