#!/usr/bin/python3

import os
from typing import List

import cv2
import numpy as np

from keras.models import Model as keras_model


def mkdir_s(path: str) -> None:
    """Create directory in specified path, if not exists.

    Parameters
    ----------
    path: str
        directory name to create

    Returns
    -------
    None

    Example
    -------
    robin.img_processing.mkdir_s(dir_name)

    """
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_in(img: np.ndarray) -> np.ndarray:
    """Normalize the input image to have all pixels in range 0 to 1.

    Parameters
    ----------
    img: numpy.ndarray
        image array to normalize

    Returns
    -------
    numpy.ndarray
        normalized image array

    Example
    -------
    robin.img_processing.normalize_in(img_array)

    """
    img = img.astype(np.float32)
    img /= 256.0
    img -= 0.5
    return img


def normalize_gt(img: np.ndarray) -> np.ndarray:
    """Normalize the gt image to have all pixels in range 0 to 1.

    Parameters
    ----------
    img: numpy.ndarray
        image array to normalize

    Returns
    -------
    numpy.ndarray
        normalized image array

    Example
    -------
    robin.img_processing.normalize_gt(img_array)

    """
    img = img.astype(np.float32)
    img /= 255.0
    return img


def add_border(
    img: np.ndarray, size_x: int = 128, size_y: int = 128
) -> (np.ndarray, int, int):
    """Add border to image,
    so it will divide window sizes: size_x and size_y

    Parameters
    ----------
    img: numpy.ndarray
        image array to add border
    size_x: int
        width for image part (deafult is `128`).
    size_y: int
        height for image part (deafult is `128`).

    Returns
    -------
    numpy.ndarray
        image array with border
    int
        border value on width
    int border value on height

    Example
    -------
    robin.img_processing.add_border(img_array, 128, 128)

    """
    max_y, max_x = img.shape[:2]
    border_y = 0
    if max_y % size_y != 0:
        border_y = (size_y - (max_y % size_y) + 1) // 2
        img = cv2.copyMakeBorder(
            img,
            border_y,
            border_y,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    border_x = 0
    if max_x % size_x != 0:
        border_x = (size_x - (max_x % size_x) + 1) // 2
        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            border_x,
            border_x,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    return img, border_y, border_x


def split_img(
    img: np.ndarray,
    size_x: int = 128,
    size_y: int = 128
) -> List[np.ndarray]:
    """Split image to parts (little images).

    Parameters
    ----------
    img: numpy.ndarray
        image array to split
    size_x: int
        width for image part (deafult is `128`).
    size_y: int
        height for image part (deafult is `128`).

    Returns
    -------
    List[numpy.ndarray]
        list of numpy.ndarray of image parts

    Notes
    -----
    Walk through the whole image by the window of size size_x * size_y
    without overlays and save all parts in list.
    Images sizes should divide window sizes.

    Example
    -------
    robin.img_processing.split_img(img_array, 128, 128)

    """
    max_y, max_x = img.shape[:2]
    parts = []
    curr_y = 0
    # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            parts.append(
                img[curr_y: curr_y + size_y, curr_x: curr_x + size_x]
            )
            curr_x += size_x
        curr_y += size_y
    return parts


def combine_imgs(imgs: List[np.ndarray], max_y: int, max_x: int) -> np.ndarray:
    """Combine image parts to one big image.

    Parameters
    ----------
    img: List[numpy.ndarray]
        list image arraies to combine
    max_y: int
        width for image part (deafult is `128`).
    max_x: int
        height for image part (deafult is `128`).

    Returns
    -------
    numpy.ndarray
        numpy.ndarray of combined image

    Notes
    -----
    Walk through list of images and create from them one big image
    with sizes max_x * max_y.
    If border_x and border_y are non-zero,
    they will be removed from created image.
    The list of images should contain data in the following order:
    from left to right, from top to bottom.

    Example
    -------
    robin.img_processing.combine_imgs(img_array_list, 128, 128)

    """
    img = np.zeros((max_y, max_x), np.float)
    size_y, size_x = imgs[0].shape
    curr_y = 0
    i = 0
    # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            try:
                img[curr_y: curr_y + size_y, curr_x: curr_x + size_x] = imgs[i]
            except Exception:
                i -= 1
            i += 1
            curr_x += size_x
        curr_y += size_y
    return img


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """Apply bilateral filter to image.

    Parameters
    ----------
    img: numpy.ndarray
        image array to preprocess

    Returns
    -------
    numpy.ndarray
        image array after preprocessing

    Example
    -------
    robin.img_preprocessing.preprocess_img(img_array)

    """
    # img = cv2.bilateralFilter(img, 5, 50, 50) TODO: change parameters.
    return img


def process_with_robin(
    img: np.ndarray, model: keras_model, batchsize: int
) -> np.ndarray:
    """Split image to 128x128 parts and run U-net for every part.

    Parameters
    ----------
    img: numpy.ndarray
        image array to preprocess
    model: keras_model
        keras model
    batchsize: int
        batchsize to use with the model

    Returns
    -------
    numpy.ndarray
        image array after preprocessing

    See Also
    --------
    add_border(), normalize_in(), split_img(), combine_imgs()

    Example
    -------
    robin.img_preprocessing.process_with_robin(process_unet_img, model)

    """
    img, border_y, border_x = add_border(img)
    img = normalize_in(img)
    parts = split_img(img)
    parts = np.array(parts)
    parts.shape = (parts.shape[0], parts.shape[1], parts.shape[2], 1)
    parts = model.predict(parts, batchsize, verbose=1)
    tmp = []
    for part in parts:
        part.shape = (128, 128)
        tmp.append(part)
    parts = tmp
    img = combine_imgs(parts, img.shape[0], img.shape[1])
    img = img[
        border_y: img.shape[0] - border_y,
        border_x: img.shape[1] - border_x
        ]
    img = img * 255.0
    img = img.astype(np.uint8)
    return img


def postprocess_img(img: np.ndarray) -> np.ndarray:
    """Apply Otsu threshold to image.

    Parameters
    ----------
    img: numpy.ndarray
        image array to postprocess

    Returns
    -------
    numpy.ndarray
        postprocessed image array

    Example
    -------
    robin.img_preprocessing.postprocess_img(img_array)

    """
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def binarize_img(
    img: np.ndarray,
    model: keras_model,
    batchsize: int = 2
) -> np.ndarray:
    """Binarize image, using U-net, Otsu, bottom-hat transform etc.

    Parameters
    ----------
    img: numpy.ndarray
        image array to preprocess
    model: keras_model
        keras model
    batchsize: int, optional
        batchsize to use with the model (default is `2`)

    Returns
    -------
    numpy.ndarray
        image array after binarizing

    See Also
    --------
    preprocess_img(), process_with_robin(), postprocess_img()

    Example
    -------
    robin.img_preprocessing.binarize_img(process_unet_img, model)

    """
    img = preprocess_img(img)
    img = process_with_robin(img, model, batchsize)
    img = postprocess_img(img)
    return img
