from robin import img_processing
import numpy as np


def test_img_processing():
    assert img_processing.normalize_gt() == np.array
