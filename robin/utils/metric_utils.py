#!/usr/bin/python3

from tensorflow import Tensor
from keras import backend as K


def dice_coef(y_true: Tensor, y_pred: Tensor) -> float:
    """Count Sorensen-Dice coefficient for output and ground-truth image.

    Parameters
    ----------
    y_true: Tensor
        tensor of true pixel values
    y_pred: Tensor
        tensor of predicted pixel values

    Returns
    -------
    float
        dice coefficient calculated on predicted and input class values.

    Example
    -------
    robin.utils.metric_utils.dice_coef(y_true, y_pred)

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + 1.0) / (
        K.sum(y_true_f)
        + K.sum(y_pred_f)
        + 1.0)
    return dice


def dice_coef_loss(y_true: Tensor, y_pred: Tensor) -> float:
    """loss of Sorensen-Dice coefficient for output and ground-truth image.

    Parameters
    ----------
    y_true: Tensor
        tensor of true pixel values
    y_pred: Tensor
        tensor of predicted pixel values

    Returns
    -------
    float
        dice loss calculated from dice coefficient.

    See Also
    --------
    dice_coef()

    Example
    -------
    robin.utils.metric_utils.dice_coef_loss(y_true, y_pred)

    """
    return 1 - dice_coef(y_true, y_pred)


def jacard_coef(y_true: Tensor, y_pred: Tensor) -> float:
    """Count Jaccard coefficient for output and ground-truth image.

    Parameters
    ----------
    y_true: Tensor
        tensor of true pixel values
    y_pred: Tensor
        tensor of predicted pixel values

    Returns
    -------
    float
        Jaccard coefficient calculated on predicted and input class values.

    Example
    -------
    robin.utils.metric_utils.jacard_coef(y_true, y_pred)

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0
    )


def jacard_coef_loss(y_true: Tensor, y_pred: Tensor) -> float:
    """Count loss of Jaccard coefficient for output and ground-truth image.

    Parameters
    ----------
    y_true: Tensor
        tensor of true pixel values
    y_pred: Tensor
        tensor of predicted pixel values

    Returns
    -------
    float
        Jaccard loss calculated from Jaccard coefficient.

    See Also
    --------
    jacard_coef()

    Example
    -------
    robin.utils.metric_utils.jacard_coef_loss(y_true, y_pred)

    """
    return 1 - jacard_coef(y_true, y_pred)
