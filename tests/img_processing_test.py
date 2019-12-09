from __future__ import print_function

from robin import img_processing
import numpy as np
import cv2


def test_img_processing():
    img = cv2.imread('test_data/input_imgs/03_in.png')
    out_img = img_processing.normalize_gt(img)
    assert(out_img, np.array)
