from data_processing.xray_reader import XRay
from data_processing.mask_functions import mask2rle  # Do I need to translate the image?

def load_data(train_paths):
    x_train = []
    y_train = []

    for image_path in train_paths:
        xray = XRay(image_path = image_path)
        x_train.append(xray.scan)
        y_train.append(xray.mask)

    return x_train, y_train


if __name__ == "__main__":
    data_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/raw_data/input/train/images/1024/"
    train_paths = sorted(glob(data_path + "/dicom/*.png"))

    # 0. Load data
    x_train, y_train = load_data(train_paths)

    # 1. Load architecture

    # 2. Configure model

    # 3. Train
    
    