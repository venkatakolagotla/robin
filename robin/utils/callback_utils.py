#!/usr/bin/python3

import os
import cv2
import imageio
import numpy as np

from alt_model_checkpoint import AltModelCheckpoint

from keras.models import Model as keras_model
from keras.callbacks import TensorBoard, Callback, EarlyStopping

from .img_processing_utils import (
    binarize_img,
    mkdir_s
)


class Visualisation(Callback):
    """Custom Keras callback for visualising training through GIFs."""

    def __init__(
        self,
        dir_name: str = "visualisation",
        batchsize: int = 20,
        monitor: str = "val_loss",
        save_best_epochs_only: bool = False,
        mode: str = "min",
    ):
        super(Visualisation, self).__init__()
        self.dir_name = dir_name
        self.batchsize = batchsize
        self.epoch_number = 0
        self.fnames = os.listdir(self.dir_name)
        for fname in self.fnames:
            mkdir_s(
                os.path.join(
                    self.dir_name,
                    fname[: fname.rfind(".")] + "_frames")
                    )
        self.monitor = monitor
        self.save_best_epochs_only = save_best_epochs_only
        self.mode = mode
        self.curr_metric = None

    def on_train_end(self, logs=None):
        for fname in self.fnames:
            frames = []
            for frame_name in sorted(
                os.listdir(
                    os.path.join(
                        self.dir_name,
                        fname[: fname.rfind(".")] + "_frames")
                )
            ):
                frames.append(
                    imageio.imread(
                        os.path.join(
                            self.dir_name,
                            fname[: fname.rfind(".")] + "_frames",
                            frame_name,
                        )
                    )
                )
            imageio.mimsave(
                os.path.join(
                    self.dir_name,
                    fname[: fname.rfind(".")] + ".gif"),
                frames,
                format="GIF",
                duration=0.5,
            )
            # rmtree(os.path.join(
            #     self.dir_name,
            #     fname[:fname.rfind('.')] + '_frames'))

    def on_epoch_end(self, epoch, logs):
        self.epoch_number += 1
        if (not self.save_best_epochs_only) or (
            (self.curr_metric is None)
            or (self.mode == "min" and logs[self.monitor] < self.curr_metric)
            or (self.mode == "max" and logs[self.monitor] > self.curr_metric)
        ):
            self.curr_metric = logs[self.monitor]
            for fname in self.fnames:
                img = cv2.imread(
                    os.path.join(self.dir_name, fname), cv2.IMREAD_GRAYSCALE
                ).astype(np.float32)
                img = binarize_img(img, self.model, self.batchsize)
                cv2.imwrite(
                    os.path.join(
                        self.dir_name,
                        fname[: fname.rfind(".")] + "_frames",
                        str(self.epoch_number) + "_out.png",
                    ),
                    img,
                )


def create_callbacks(
    model: keras_model,
    original_model: keras_model,
    debug: str,
    num_gpus: int,
    augmentate: bool,
    batchsize: int,
    vis: str,
    weights_path: str,
) -> list:
    """Create Keras callbacks for training.

    Parameters
    ----------
    model: keras_model
        keras model
    original_model: keras_model
        model to use when num_gpus > 1

    Returns
    -------
    list
        list of callbacks tu use in training.

    See Also
    --------
    main()

    Example
    -------
    robin.train.create_callbacks(model, gpu_model)

    """
    callbacks = []

    # Model checkpoint.
    if num_gpus == 1:
        model_checkpoint = AltModelCheckpoint(
            weights_path
            if debug == ""
            else os.path.join(
                debug,
                "weights",
                "weights-improvement-{epoch:02d}.hdf5"),
            model,
            monitor="val_dice_coef",
            mode="max",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )
    else:
        model_checkpoint = AltModelCheckpoint(
            weights_path
            if debug == ""
            else os.path.join(
                debug,
                "weights",
                "weights-improvement-{epoch:02d}.hdf5"),
            original_model,
            monitor="val_dice_coef",
            mode="max",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )
    callbacks.append(model_checkpoint)

    # Early stopping.
    model_early_stopping = EarlyStopping(
        monitor="val_dice_coef",
        min_delta=0.001,
        patience=20,
        verbose=1,
        mode="max"
    )
    callbacks.append(model_early_stopping)

    # Tensorboard logs.
    if debug != "":
        mkdir_s(debug)
        mkdir_s(os.path.join(debug, "weights"))
        mkdir_s(os.path.join(debug, "logs"))
        model_tensorboard = TensorBoard(
            log_dir=os.path.join(debug, "logs"),
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        callbacks.append(model_tensorboard)

    # Training visualisation.
    if vis != "":
        model_visualisation = Visualisation(
            dir_name=vis,
            batchsize=batchsize,
            monitor="val_dice_coef",
            save_best_epochs_only=True,
            mode="max",
        )
        callbacks.append(model_visualisation)

    return callbacks
