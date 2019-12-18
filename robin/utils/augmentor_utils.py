#!/usr/bin/python3

from typing import List

import cv2
import PIL
import numpy as np
from Augmentor.Operations import Operation


class GaussianNoiseAugmentor(Operation):
    """Gaussian Noise in Augmentor format.

    Parameters
    ----------
    probability: float
        probability of operation being performed
    mean: float
        mean of the pixels
    sigma: float
        standard deviation of the pixels

    """
    def __init__(self, probability: float, mean: float, sigma: float):
        Operation.__init__(self, probability)
        self.mean = mean
        self.sigma = sigma

    def __gaussian_noise__(self, image: np.ndarray) -> np.ndarray:
        """Apply gaussian noise to the image.

        Parameters
        ----------
        image: numpy.ndarray
            image to perform opetation

        Returns
        -------
        numpy.ndarray
            image array after operation is performed

        """
        img = np.array(image).astype(np.int16)
        tmp = np.zeros(img.shape, np.int16)
        img = img + cv2.randn(tmp, self.mean, self.sigma)
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        image = PIL.Image.fromarray(img)

        return image

    def perform_operation(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply operation to a batch of images.

        Parameters
        ----------
        image: List[numpy.ndarray]
            list images to perform opetation

        Returns
        -------
        List[numpy.ndarray]
            list of image arrays after operation is performed

        """
        images = [self.__gaussian_noise__(image) for image in images]
        return images


class SaltPepperNoiseAugmentor(Operation):
    """Gaussian Noise in Augmentor format.

    Parameters
    ----------
    probability: float
        probability of operation being performed
    prop: float
        image proportion value to keep

    """
    def __init__(self, probability: float, prop: float):
        Operation.__init__(self, probability)
        self.prop = prop

    def __salt_pepper_noise__(self, image: np.ndarray) -> np.ndarray:
        """Apply salt_pepper noise to the image.

        Parameters
        ----------
        image: numpy.ndarray
            image to perform opetation

        Returns
        -------
        numpy.ndarray
            image array after operation is performed

        """
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

    def perform_operation(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply operation to a batch of images.

        Parameters
        ----------
        image: List[numpy.ndarray]
            list of images to perform opetation

        Returns
        -------
        List[numpy.ndarray]
            list of image arrays after operation is performed

        """
        images = [self.__salt_pepper_noise__(image) for image in images]
        return images


class InvertPartAugmentor(Operation):
    """Invert colors in Augmentor formant."""

    def __init__(self, probability):
        Operation.__init__(self, probability)

    def __invert__(self, image: np.ndarray) -> np.ndarray:
        """Apply invert operation to the image.

        Parameters
        ----------
        image: numpy.ndarray
            image to perform opetation

        Returns
        -------
        numpy.ndarray
            image array after operation is performed

        """
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

    def perform_operation(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply invert operation to a batch of images.

        Parameters
        ----------
        image: List[numpy.ndarray]
            list of images to perform opetation

        Returns
        -------
        List[numpy.ndarray]
            list of image arrays after operation is performed

        """
        images = [self.__invert__(image) for image in images]
        return images
