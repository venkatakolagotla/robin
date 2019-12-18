from __future__ import print_function
from robin import dataset


def test_split_img_overlay(in_img_array):
    output = dataset.split_img_overlay(in_img_array)
    assert(type(output[0]) == list)
    assert(type(output[1]) == int)
    assert(type(output[2]) == int)
