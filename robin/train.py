#!/usr/bin/python3

import os
import sys
import time
import random
from copy import deepcopy

import cv2
import PIL
import imageio
import numpy as np
from Augmentor import DataPipeline
from Augmentor.Operations import Operation

from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model as keras_model
from keras.callbacks import TensorBoard, Callback, EarlyStopping
from keras.utils import multi_gpu_model, Sequence
from alt_model_checkpoint import AltModelCheckpoint

from .unet import unet
from .img_processing import binarize_img, mkdir_s, normalize_gt, normalize_in


class GaussianNoiseAugmentor(Operation):
    """Gaussian Noise in Augmentor format."""

    def __init__(self, probability, mean, sigma):
        Operation.__init__(self, probability)
        self.mean = mean
        self.sigma = sigma

    def __gaussian_noise__(self, image):
        img = np.array(image).astype(np.int16)
        tmp = np.zeros(img.shape, np.int16)
        img = img + cv2.randn(tmp, self.mean, self.sigma)
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        image = PIL.Image.fromarray(img)

        return image

    def perform_operation(self, images):
        images = [self.__gaussian_noise__(image) for image in images]
        return images


class SaltPepperNoiseAugmentor(Operation):
    """Gaussian Noise in Augmentor format."""

    def __init__(self, probability, prop):
        Operation.__init__(self, probability)
        self.prop = prop

    def __salt_pepper_noise__(self, image):
        img = np.array(image).astype(np.uint8)
        h = img.shape[0]
        w = img.shape[1]
        n = int(h * w * self.prop)
        for i in range(n // 2):
            # Salt.
            curr_y = int(np.random.randint(0, h))
            curr_x = int(np.random.randint(0, w))
            img[curr_y, curr_x] = 255
        for i in range(n // 2):
            # Pepper.
            curr_y = int(np.random.randint(0, h))
            curr_x = int(np.random.randint(0, w))
            img[curr_y, curr_x] = 0
        image = PIL.Image.fromarray(img)

        return image

    def perform_operation(self, images):
        images = [self.__salt_pepper_noise__(image) for image in images]
        return images


class InvertPartAugmentor(Operation):
    """Invert colors in Augmentor formant."""

    def __init__(self, probability):
        Operation.__init__(self, probability)

    def __invert__(self, image):
        img = np.array(image).astype(np.uint8)
        h = img.shape[0]
        w = img.shape[1]
        y_begin = int(np.random.randint(0, h))
        x_begin = int(np.random.randint(0, w))
        y_add = int(np.random.randint(0, h - y_begin))
        x_add = int(np.random.randint(0, w - x_begin))
        for i in range(y_begin, y_begin + y_add):
            for j in range(x_begin, x_begin + x_add):
                img[i][j] = 255 - img[i][j]
        image = PIL.Image.fromarray(img)

        return image

    def perform_operation(self, images):
        images = [self.__invert__(image) for image in images]
        return images


class ParallelDataGenerator(Sequence):
    """Generate images for training/validation/testing (parallel version)."""

    def __init__(
        self,
        fnames_in,
        fnames_gt,
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

    def __apply_augmentation__(self, p):
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

    def augmentate_batch(self, imgs_in, imgs_gt):
        """Generate ordered augmented batch of images, using Augmentor"""

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
        # Creating numpy arrays with images.
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


def dice_coef(y_true: keras_model, y_pred: keras_model) -> float:
    """Count Sorensen-Dice coefficient for output and ground-truth image.

    Parameters
    ----------
    y_true: keras_model
        trained keras model
    y_pred: keras_model
        trained multi gpu model

    Returns
    -------
    float
        dice coefficient calculated on predicted and input class values.

    Example
    -------
    robin.train.dice_coef(y_true, y_pred)

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + 1.0) / (
        K.sum(y_true_f)
        + K.sum(y_pred_f)
        + 1.0)
    return dice


def dice_coef_loss(y_true, y_pred):
    """loss of Sorensen-Dice coefficient for output and ground-truth image.

    Parameters
    ----------
    y_true: keras_model
        trained keras model
    y_pred: keras_model
        trained multi gpu model

    Returns
    -------
    float
        dice loss calculated from dice coefficient.

    See Also
    --------
    dice_coef()

    Example
    -------
    robin.train.dice_coef_loss(y_true, y_pred)

    """
    return 1 - dice_coef(y_true, y_pred)


def jacard_coef(y_true, y_pred):
    """Count Jaccard coefficient for output and ground-truth image.

    Parameters
    ----------
    y_true: keras_model
        trained keras model
    y_pred: keras_model
        trained multi gpu model

    Returns
    -------
    float
        Jaccard coefficient calculated on predicted and input class values.

    Example
    -------
    robin.train.jacard_coef(y_true, y_pred)

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0
    )


def jacard_coef_loss(y_true, y_pred):
    """Count loss of Jaccard coefficient for output and ground-truth image.

    Parameters
    ----------
    y_true: keras_model
        trained keras model
    y_pred: keras_model
        trained multi gpu model

    Returns
    -------
    float
        Jaccard loss calculated from Jaccard coefficient.

    See Also
    --------
    jacard_coef()

    Example
    -------
    robin.train.jacard_coef_loss(y_true, y_pred)

    """
    return 1 - jacard_coef(y_true, y_pred)


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
    input: str
        input dir with in and gt sub folders to train
        (default is os.path.join(".", "input")).
    vis: str
        dir with image to use for train visualization
        (default is os.path.join(".", "vis")).
    debug: str
        path to save training logs
        (default is os.path.join(".", "train_logs")).
    epochs: int
        number of epocs to train robin (default is 1).
    batchsize: int
        batchsize to train robin (default is 32).
    augmentate: bool
        argumentate the original images for training robin
        (default is True)
    train_split: int
        train dataset split percentage (default is 80).
    val_split: int
        validation dataset split percentage (default is 10).
    test_split: int
        train dataset split percentage (default is 10).
    weights_path: str
        path to save final weights
        (default is os.path.join(".", "bin_weights.hdf5")).
    num_gpus: int
        number of gpus to use for training robin (default is 1)
    extraprocesses: int
        number of extraprocesses to use (default is 0).
    queuesize: int
        number of batches to generate in queue while training
        (default is 10).

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
    # assert (validation_generator.__len__() >= batchsize)
    # assert (test_generator.__len__() >= batchsize)
    print(validation_generator.__len__())
    print(test_generator.__len__())

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
    # Sometimes script freezes.
    sys.exit(0)


if __name__ == "__main__":
    main()
