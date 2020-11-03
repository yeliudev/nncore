# Copyright (c) Ye Liu. All rights reserved.

import cv2
import numpy as np


def imnormalize(img, mean, std):
    """
    Normalize an image with mean and std.

    Args:
        img (:obj:`np.ndarray`): image to be normalized
        mean (:obj:`np.ndarray`): the mean to be used for normalize
        std (:obj:`np.ndarray`): the std to be used for normalize

    Returns:
        img (:obj:`np.ndarray`): the normalized image
    """
    if img.dtype == np.uint8:
        img = np.float32(img)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


def imdenormalize(img, mean, std):
    """
    Denormalize an image with mean and std.

    Args:
        img (:obj:`np.ndarray`): image to be denormalized
        mean (:obj:`np.ndarray`): the mean to be used for denormalize
        std (:obj:`np.ndarray`): the std to be used for denormalize

    Returns:
        img (:obj:`np.ndarray`): the denormalized image
    """
    if img.dtype == np.uint8:
        img = np.float32(img)
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)
    cv2.add(img, mean, img)
    return img
