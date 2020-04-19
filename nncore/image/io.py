# Copyright (c) Ye Liu. All rights reserved.

import cv2

import nncore

_colorspaces = {
    'color': cv2.IMREAD_COLOR,
    'grayscale': cv2.IMREAD_GRAYSCALE,
    'unchanged': cv2.IMREAD_UNCHANGED
}


def imread(filename, flag='color', channel_order='bgr'):
    """
    Read an image from a file.

    Args:
        filename (str): name of the image file
        flag (str or int, optional): flags specifying the color type of the
            loaded image. Currently supported flags include `color`,
            `grayscale` and `unchanged`.
        channel_order (str, optional): order of the channels. Currently
            supported orders include `bgr` and `rgb`.

    Returns:
        img (ndarray): the loaded image array
    """
    if not isinstance(filename, str):
        raise TypeError(
            "filename must be a str, but got '{}'".format(filename))

    nncore.file_exist(filename, raise_error=True)

    flag = _colorspaces[flag] if isinstance(flag, str) else flag
    img = cv2.imread(filename, flag)

    if flag == cv2.IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

    return img


def imwrite(img, filename, params=None):
    """
    Write an image to a file.

    Args:
        img (ndarray): the image array to be written
        filename (str): name of the image file
        params (None or list, optional): same as opencv's :func:`imwrite`
            interface

    Returns:
        flag (bool): successful or not
    """
    return cv2.imwrite(filename, img, params)
