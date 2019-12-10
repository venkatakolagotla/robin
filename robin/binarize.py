#!/usr/bin/python3

import os
import glob
import time

import cv2
import numpy as np

from keras.optimizers import Adam

from .unet import unet
from .utils.metric_utils import dice_coef, dice_coef_loss
from .utils.img_processing_utils import binarize_img, mkdir_s


weights_path = os.path.realpath(__file__)
weights_path = weights_path.replace(
    "robin/binarize.py",
    "weights/bin_weights.hdf5")


def main(
    input: str = os.path.join(".", "input"),
    output: str = os.path.join(".", "output"),
    weights_path: str = weights_path,
    batchsize: int = 2,
) -> [np.array]:
    """Binarize images from input directory and
    write them to output directory.

    Parameters
    ----------
    input: str
        input path for images
    output: str
        output path to save images
    weights_path: str
        path to weights file
    batchsize: int
        batchsize to use in model prediction

    Retuns
    ------
    array_like
        list of binary images in np.array format

    Notes
    -----
    All input image names should be in png format "sample_1.png".
    All output image names will end with "_bin" like "sample_1_bin.png".

    Example
    -------
    robin.binarize.main('input_path', 'output_path', 2)

    """
    try:
        assert (batchsize > 0) and isinstance(batchsize, int)
    except Exception:
        print("batchsize should be > 0 and int but given {}".format(batchsize))
    try:
        assert (os.path.isdir(input))
    except Exception:
        print("Input path is not valid")

    start_time = time.time()

    fnames_in = list(glob.iglob(os.path.join(
        input,
        "**",
        "*.png*"), recursive=True))
    model = None
    if len(fnames_in) != 0:
        mkdir_s(output)
        model = unet()
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss=dice_coef_loss,
            metrics=[dice_coef])
        model.load_weights(weights_path)

    bin_img_list = []

    for fname in fnames_in:
        print("binarizing -> {0}".format(fname))
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = binarize_img(img, model, batchsize)
        cv2.imwrite(
            os.path.join(
                output,
                os.path.split(fname)[-1].replace(".png", "_bin.png")),
            img
        )
        bin_img_list.append(img)
    print("finished in {0:.2f} seconds".format(time.time() - start_time))
    return bin_img_list


if __name__ == "__main__":
    main()
