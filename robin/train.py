#!/usr/bin/python3

import os
import time
import random
from copy import deepcopy
from typing import List, Tuple

import cv2
import PIL
import numpy as np
from Augmentor import DataPipeline

from keras.optimizers import Adam
from keras.utils import multi_gpu_model, Sequence

from .unet import unet
from .utils.metric_utils import (
    dice_coef,
    dice_coef_loss,
    jacard_coef
)
from .utils.augmentor_utils import (
    GaussianNoiseAugmentor,
    InvertPartAugmentor,
    SaltPepperNoiseAugmentor
)
from .utils.img_processing_utils import (
    normalize_gt,
    normalize_in
)
from .utils.callback_utils import create_callbacks


class ParallelDataGenerator(Sequence):
    """Generate images for training/validation/testing (parallel version).

    Parameters
    ----------
    fnames_in: List[str]
        list of input images
    fnames_gt: List[str]
        list of gt images
    batch_size: int
        batch size to generate augmentations on images
    augmentate: bool
        apply augmentate to batch of images

    """
    def __init__(
        self,
        fnames_in: List[str],
        fnames_gt: List[str],
        batch_size: int,
        augmentate: bool
    ):
        self.fnames_in = deepcopy(fnames_in)
        self.fnames_gt = deepcopy(fnames_gt)
        self.batch_size = batch_size
        self.augmentate = augmentate
        self.idxs = np.array([i for i in range(len(self.fnames_in))])

    def __len__(self):
        return int(np.ceil(float(self.idxs.shape[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.idxs)

    def __apply_augmentation__(self, p: object) -> List[np.ndarray]:
        """Apply augmentation on batch of images"""
        batch = []
        for i in range(0, len(p.augmentor_images)):
            images_to_return = [
                PIL.Image.fromarray(x) for x in p.augmentor_images[i]
                ]

            for operation in p.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    images_to_return = operation.perform_operation(
                        images_to_return
                        )

            images_to_return = [np.asarray(x) for x in images_to_return]
            batch.append(images_to_return)
        return batch

    def augmentate_batch(
        self,
        imgs_in: List[np.ndarray],
        imgs_gt: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate ordered augmented batch of images, using Augmentor

        Parameters
        ----------
        imgs_in: List[numpy.ndarray]
            list of input images as array
        imgs_gt: List[numpy.ndarray]
            list of gt image as array
        Returns
        -------
        Tuple[List[numpy.ndarray], List[numpy.ndarray]]
            List of input images after applying augmentation
            List of gt images after applying augmentation

        """
        # Non-Linear transformations.
        imgs = [[imgs_in[i], imgs_gt[i]] for i in range(len(imgs_in))]
        p = DataPipeline(imgs)
        p.random_distortion(0.5, 6, 6, 4)
        # Linear transformations.
        # p.rotate(0.75, 15, 15)
        p.shear(0.75, 10.0, 10.0)
        p.zoom(0.75, 1.0, 1.2)
        p.skew(0.75, 0.75)
        imgs = self.__apply_augmentation__(p)
        imgs_in = [p[0] for p in imgs]
        imgs_gt = [p[1] for p in imgs]

        # Noise transformations.
        p = DataPipeline([[img] for img in imgs_in])
        gaussian_noise = GaussianNoiseAugmentor(0.25, 0, 10)
        p.add_operation(gaussian_noise)
        salt_pepper_noise = SaltPepperNoiseAugmentor(0.25, 0.005)
        p.add_operation(salt_pepper_noise)
        # Brightness transformation.
        p.random_brightness(0.75, 0.5, 1.5)
        p.random_contrast(0.75, 0.5, 1.5)
        # Colors invertion.
        invert = InvertPartAugmentor(0.25)
        p.add_operation(invert)
        p.invert(0.5)
        imgs_in = self.__apply_augmentation__(p)
        imgs_in = [p[0] for p in imgs_in]

        return imgs_in, imgs_gt

    def __getitem__(self, idx):
        """Creates numpy arrays with images."""
        start = idx * self.batch_size
        stop = start + self.batch_size
        if stop >= self.idxs.shape[0]:
            stop = self.idxs.shape[0]

        imgs_in = []
        imgs_gt = []
        for i in range(start, stop):
            imgs_in.append(
                cv2.imread(self.fnames_in[self.idxs[i]], cv2.IMREAD_GRAYSCALE)
            )
            imgs_gt.append(
                cv2.imread(self.fnames_gt[self.idxs[i]], cv2.IMREAD_GRAYSCALE)
            )

        # Applying augmentations.
        if self.augmentate:
            imgs_in, imgs_gt = self.augmentate_batch(imgs_in, imgs_gt)

        """
        # Debug.
        for i in range(len(imgs_in)):
            cv2.imshow('in_' + str(i), imgs_in[i])
            cv2.imshow('gt_' + str(i), imgs_gt[i])
            cv2.waitKey(0)
        """

        # Normalization.
        imgs_in = np.array([normalize_in(img) for img in imgs_in])
        imgs_in.shape = (
            imgs_in.shape[0],
            imgs_in.shape[1],
            imgs_in.shape[2],
            1
            )
        imgs_gt = np.array([normalize_gt(img) for img in imgs_gt])
        imgs_gt.shape = (
            imgs_gt.shape[0],
            imgs_gt.shape[1],
            imgs_gt.shape[2],
            1
            )

        return imgs_in, imgs_gt


def main(
    input: str = os.path.join(".", "input"),
    vis: str = os.path.join(".", "vis"),
    debug: str = os.path.join(".", "train_logs"),
    epochs: int = 1,
    batchsize: int = 32,
    augmentate: bool = True,
    train_split: int = 80,
    val_split: int = 10,
    test_split: int = 10,
    weights_path: str = os.path.join(".", "bin_weights.hdf5"),
    num_gpus: int = 1,
    extraprocesses: int = 0,
    queuesize: int = 10,
):
    """Train U-net with pairs of train and ground-truth images.

    Parameters
    ----------
    input: str, optional
        input dir with in and gt sub folders to train
        (default is os.path.join(".", "input")).
    vis: str, optional
        dir with image to use for train visualization
        (default is os.path.join(".", "vis")).
    debug: str, optional
        path to save training logs
        (default is os.path.join(".", "train_logs")).
    epochs: int, optional
        number of epocs to train robin (default is `1`).
    batchsize: int, optional
        batchsize to train robin (default is `32`).
    augmentate: bool, optional
        argumentate the original images for training robin
        (default is `True`)
    train_split: int, optional
        train dataset split percentage (default is `80`).
    val_split: int, optional
        validation dataset split percentage (default is `10`).
    test_split: int, optional
        train dataset split percentage (default is `10`).
    weights_path: str, optional
        path to save final weights
        (default is os.path.join(".", "bin_weights.hdf5")).
    num_gpus: int, optional
        number of gpus to use for training robin (default is `1`)
    extraprocesses: int, optional
        number of extraprocesses to use (default is `0`).
    queuesize: int, optional
        number of batches to generate in queue while training
        (default is `10`).

    Retunrs
    -------
    None

    Notes
    -----
    All train images should be in "in" directory.
    All ground-truth images should be in "gt" directory.

    Example
    -------
    robin.train.main(input, vis, logs_dir, 2, 4)

    """
    assert epochs > 0
    assert batchsize > 0

    assert train_split >= 0
    assert val_split >= 0
    assert test_split >= 0

    assert num_gpus >= 1
    assert extraprocesses >= 0
    assert queuesize >= 0

    start_time = time.time()
    np.random.seed()

    # Creating data for training, validation and testing.
    fnames_in = [
        os.path.join(input, "in", str(i) + "_in.png")
        for i in range(len(os.listdir(os.path.join(input, "in"))))
    ]
    fnames_gt = [
        os.path.join(input, "gt", str(i) + "_gt.png")
        for i in range(len(os.listdir(os.path.join(input, "gt"))))
    ]
    assert len(fnames_in) == len(fnames_gt)
    n = len(fnames_in)

    train_start = 0

    train_stop = int(n * (train_split / 100))
    train_in = fnames_in[train_start:train_stop]
    train_gt = fnames_gt[train_start:train_stop]
    train_generator = ParallelDataGenerator(
        train_in,
        train_gt,
        batchsize,
        augmentate
        )

    validation_start = train_stop
    validation_stop = validation_start + int(n * (val_split / 100))
    validation_in = fnames_in[validation_start:validation_stop]
    validation_gt = fnames_gt[validation_start:validation_stop]
    validation_generator = ParallelDataGenerator(
        validation_in, validation_gt, batchsize, augmentate
    )

    test_start = validation_stop
    test_stop = n
    test_in = fnames_in[test_start:test_stop]
    test_gt = fnames_gt[test_start:test_stop]
    test_generator = ParallelDataGenerator(
        test_in,
        test_gt,
        batchsize,
        augmentate
        )

    # check if validation steps are more than batch size or not
    assert (validation_generator.__len__() >= batchsize)
    assert (test_generator.__len__() >= batchsize)

    # Creating model.
    original_model = unet()
    if num_gpus == 1:
        model = original_model
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss=dice_coef_loss,
            metrics=[dice_coef, jacard_coef, "accuracy"],
        )
        model.summary()
    else:
        model = multi_gpu_model(original_model, gpus=num_gpus)
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss=dice_coef_loss,
            metrics=[dice_coef, jacard_coef, "accuracy"],
        )
        model.summary()
    callbacks = create_callbacks(
        model,
        original_model,
        debug,
        num_gpus,
        augmentate,
        batchsize,
        vis,
        weights_path
    )

    # Running training, validation and testing.
    if extraprocesses == 0:
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.__len__(),
            # Compatibility with old Keras versions.
            validation_data=validation_generator,
            validation_steps=validation_generator.__len__(),
            # Compatibility with old Keras versions.
            epochs=epochs,
            shuffle=True,
            callbacks=callbacks,
            use_multiprocessing=False,
            workers=0,
            max_queue_size=queuesize,
            verbose=1,
        )
        metrics = model.evaluate_generator(
            generator=test_generator,
            use_multiprocessing=False,
            workers=0,
            max_queue_size=queuesize,
            verbose=1,
        )
    else:
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.__len__(),
            # Compatibility with old Keras versions.
            validation_data=validation_generator,
            validation_steps=validation_generator.__len__(),
            # Compatibility with old Keras versions.
            epochs=epochs,
            shuffle=True,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=extraprocesses,
            max_queue_size=queuesize,
            verbose=1,
        )
        metrics = model.evaluate_generator(
            generator=test_generator,
            use_multiprocessing=True,
            workers=extraprocesses,
            max_queue_size=queuesize,
            verbose=1,
        )

    print()
    print("total:")
    print("test_loss:       {0:.4f}".format(metrics[0]))
    print("test_dice_coef:  {0:.4f}".format(metrics[1]))
    print("test_jacar_coef: {0:.4f}".format(metrics[2]))
    print("test_accuracy:   {0:.4f}".format(metrics[3]))

    # Saving model.
    if debug != "":
        model.save_weights(weights_path)
    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
