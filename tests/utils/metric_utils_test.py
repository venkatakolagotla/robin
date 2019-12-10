from robin.utils import metric_utils
import numpy as np
import cv2
import tensorflow as tf

img_gt = cv2.imread("test_data/01_in_gt.png")
img_gt = np.asarray(img_gt, np.float32)
img_bin = cv2.imread("test_data/01_in_bin.png")
img_bin = np.asarray(img_bin, np.float32)
y_true = tf.convert_to_tensor(img_gt, np.float32)
y_pred = tf.convert_to_tensor(img_bin, np.float32)


def test_dice_coef():
    # dice coefficient test
    dice_coef = metric_utils.dice_coef(y_true, y_pred)
    assert dice_coef.dtype == float


def test_dice_coef_loss():
    # dice loss test
    dice_loss = metric_utils.dice_coef_loss(y_true, y_pred)
    assert dice_loss.dtype == float


def test_jaccard_coef():
    # jaccard coefficient test
    jaccard_coef = metric_utils.jacard_coef(y_true, y_pred)
    assert jaccard_coef.dtype == float


def test_jaccard_coef_loss():
    # jacccard loss test
    jaccard_loss = metric_utils.jacard_coef_loss(y_true, y_pred)
    assert jaccard_loss.dtype == float
