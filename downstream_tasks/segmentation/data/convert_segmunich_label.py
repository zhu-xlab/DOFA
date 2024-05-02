#
from mmengine import list_dir_or_file
import tifffile as tif
import numpy as np
import os
import cv2
import tqdm

data_root = "/home/zhitong/Datasets/SegMunich/ann_dir"

for img_name in tqdm.tqdm(list_dir_or_file("./val", suffix=".tif", list_dir=False)):
    new_img_name = os.path.join(data_root, "val", img_name)
    target = tif.imread(new_img_name)
    target[target == 21] = 1
    target[target == 22] = 2
    target[target == 23] = 3
    target[target == 31] = 4
    target[target == 32] = 6
    target[target == 33] = 7
    target[target == 41] = 8
    target[target == 13] = 9
    target[target == 14] = 10
    save_label_name = new_img_name.replace("tif","png")
    cv2.imwrite(save_label_name, target)
    #print(np.unique(target))