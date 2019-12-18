from __future__ import print_function

from robin.utils import img_processing_utils
import numpy as np


def test_normalize_gt(in_img_array):
    out_img = img_processing_utils.normalize_gt(in_img_array)
    assert type(out_img) == np.ndarray
