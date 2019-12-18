import os
import pytest

import cv2
import numpy as np
import tensorflow as tf


@pytest.fixture(scope="module")
def base_path():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def data_path(base_path):
    base_data_path = base_path + "/data"
    return base_data_path


@pytest.fixture(scope="module")
def in_img_path(data_path):
    img_in_path = data_path + "/01_in.png"
    return img_in_path


@pytest.fixture(scope="module")
def gt_img_path(data_path):
    img_gt_path = data_path + "/01_gt.png"
    return img_gt_path


@pytest.fixture(scope="module")
def bin_img_path(data_path):
    img_bin_path = data_path + "/01_bin_in.png"
    return img_bin_path


@pytest.fixture(scope="module")
def in_img_array(in_img_path):
    img_in_array = cv2.imread(in_img_path)
    return img_in_array


@pytest.fixture(scope="module")
def y_true(gt_img_path):
    img_gt = cv2.imread(gt_img_path)
    img_gt = np.asarray(img_gt, np.float32)
    y_true = tf.convert_to_tensor(img_gt, np.float32)
    return y_true


@pytest.fixture(scope="module")
def y_pred(bin_img_path):
    img_bin = cv2.imread(bin_img_path)
    img_bin = np.asarray(img_bin, np.float32)
    y_pred = tf.convert_to_tensor(img_bin, np.float32)
    return y_pred
