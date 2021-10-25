# Copyright (c) Ye Liu. All rights reserved.

import cv2

_INTERP_CODES = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def imresize(img, size, interpolation='bilinear', return_scale=False):
    """
    Resize an image to a given size.

    Args:
        img (:obj:`np.ndarray`): The input image.
        size (tuple[int]): The target size in the form of ``(width, height)``.
        interpolation (str | int, optional): Interpolation method. Currently
            supported methods include ``nearest``, ``bilinear``, ``bicubic``,
            ``area``, and ``lanczos``. Default: ``bilinear``.
        return_scale (bool, optional): Whether to return ``w_scale`` and
            ``h_scale``. Default: ``False``.

    Returns:
        :obj:`np.ndarray` | tuple: The resized image (and scales).
    """
    out_img = cv2.resize(img, size, interpolation=_INTERP_CODES[interpolation])

    if return_scale:
        h, w = img.shape[:2]

        w_scale = size[0] / w
        h_scale = size[1] / h

        return out_img, w_scale, h_scale
    else:
        return out_img


def imresize_like(img, target, interpolation='bilinear', return_scale=False):
    """
    Resize an image to the same size of a given image.

    Args:
        img (:obj:`np.ndarray`): The input image.
        target (:obj:`np.ndarray`): The target image.
        interpolation (str | int, optional): Interpolation method. Currently
            supported methods include ``nearest``, ``bilinear``, ``bicubic``,
            ``area``, and ``lanczos``. Default: ``bilinear``.
        return_scale (bool, optional): Whether to return ``w_scale`` and
            ``h_scale``. Default: ``False``.

    Returns:
        :obj:`np.ndarray` | tuple: The resized image (and scales).
    """
    return imresize(
        img,
        target.shape[:2],
        interpolation=interpolation,
        return_scale=return_scale)


def rescale_size(size, scale, return_scale=False):
    """
    Compute the new size to be rescaled to.

    Args:
        size (tuple[int]): The original size in the form of
            ``(width, height)``.
        scale (int | tuple[int]): The scaling factor or the maximum size. If
            it is a number, the image will be rescaled by this factor. When it
            is a tuple containing 2 numbers, the image will be rescaled as
            large as possible within the scale. In this case, ``-1`` means
            infinity.
        return_scale (bool, optional): Whether to return the scaling factor.
            Default: ``False``.

    Returns:
        :obj:`np.ndarray` | tuple: The new size (and scaling factor).
    """
    w, h = size

    if isinstance(scale, (float, int)):
        scale_factor = scale
    elif isinstance(scale, tuple):
        if -1 in scale:
            max_s_edge = max(scale)
            scale_factor = max_s_edge / min(h, w)
        else:
            max_l_edge = max(scale)
            max_s_edge = min(scale)
            scale_factor = min(max_l_edge / max(h, w), max_s_edge / min(h, w))
    else:
        raise TypeError(
            "'scale must be a number or tuple of int, but got '{}'".format(
                type(scale)))

    new_size = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(img, scale, interpolation='bilinear', return_scale=False):
    """
    Resize an image while keeping the aspect ratio.

    Args:
        img (:obj:`np.ndarray`): The input image.
        scale (int | tuple[int]): The scaling factor or the maximum size. If
            it is a number, the image will be rescaled by this factor. When it
            is a tuple containing 2 numbers, the image will be rescaled as
            large as possible within the scale. In this case, ``-1`` means
            infinity.
        interpolation (str | int, optional): Interpolation method. Currently
            supported methods include ``nearest``, ``bilinear``, ``bicubic``,
            ``area``, and ``lanczos``. Default: ``bilinear``.
        return_scale (bool, optional): Whether to return the scaling factor.
            Default: ``False``.

    Returns:
        :obj:`np.ndarray` | tuple: The resized image (and scaling factor).
    """
    h, w = img.shape[:2]
    size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    out_img = imresize(img, size, interpolation=interpolation)

    if return_scale:
        return out_img, scale_factor
    else:
        return out_img
