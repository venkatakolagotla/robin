from __future__ import print_function
from robin import dataset
import cv2


def test_dataset():
    img = cv2.imread('test_data/input_imgs/02_in.png')
    print("---------------------------------", type(img))
    output = dataset.split_img_overlay(img)
    assert(type(output[0]) == list)
    assert(type(output[1]) == int)
    assert(type(output[2]) == int)
