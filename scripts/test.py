import pandas as pd
from tqdm import tqdm
import cv2
import os
import numpy as np
# Use cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import model_from_json
from data_processing.xray_reader import XRay

def recover_logged_model(weights_path):
    weights_name = weights_path.split("/")[-1]
    full_model_path = weights_path.replace("/" + weights_name, "")
    json_file = open(full_model_path + "/architecture.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")
    return loaded_model

xray = XRay()
mod = recover_logged_model(
    weights_path = "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/models/2019-08-03_17:38:49/weights-2.000000-0.7825.hdf5")

sample_df = pd.read_csv(
    "/home/oleguer/projects/kaggle_Pneumothorax-Segmentation/kaggle_provided_data/sample_submission.csv")

# List of image ids
# masks_ = sample_df.groupby('ImageId')['ImageId'].count().reset_index(name='N')
# masks_ = masks_.loc[masks_.N > 1].ImageId.values

# Remove duplicates
sample_df = sample_df.drop_duplicates(
    'ImageId', keep='last').reset_index(drop=True)

# print(masks_)

sublist = []
counter = 0
threshold = 0.3
for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    image_id = row['ImageId']
    # if image_id in masks_:
    img_path = os.path.join(
        '/media/oleguer/Extenció/FEINA/projectes/pneumotorax/input/test/images/512/dicom/', image_id + '.png')

    scan = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.
    scan = scan.reshape((-1, scan.shape[0], scan.shape[1], 1))

    mask = mod.predict(scan)
    scan = np.array(scan[0,:,:,0]*255, dtype=np.uint8)
    mask = np.array(mask[0,:,:,0]*255, dtype=np.uint8)

    if np.amax(mask) > 0:
        xray.plot_composition(scan, mask)


#         if len(result["masks"]) > 0:
#             counter += 1
#             mask_added = 0
#             for ppx in range(len(result["masks"])):
#                 if result["scores"][ppx] >= threshold:
#                     mask_added += 1
#                     res = transforms.ToPILImage()(
#                         result["masks"][ppx].permute(1, 2, 0).cpu().numpy())
#                     res = np.asarray(res.resize(
#                         (width, height), resample=Image.BILINEAR))
#                     res = (res[:, :] * 255. > 127).astype(np.uint8).T
#                     rle = mask_to_rle(res, width, height)
#                     sublist.append([image_id, rle])
#             if mask_added == 0:
#                 rle = " -1"
#                 sublist.append([image_id, rle])
#         else:
#             rle = " -1"
#             sublist.append([image_id, rle])
#     else:
#         rle = " -1"
#         sublist.append([image_id, rle])

# submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
# submission_df.to_csv("submission.csv", index=False)
# print(counter)
