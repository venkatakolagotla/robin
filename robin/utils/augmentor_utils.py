#!/usr/bin/python3

import cv2
import PIL
import numpy as np
from Augmentor.Operations import Operation


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
