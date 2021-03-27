# Copyright (c) Ye Liu. All rights reserved.

import cv2


def bgr2gray(img, keep_dim=False):
    """
    Convert a BGR image to a grayscale image.

    Args:
        img (:obj:`np.ndarray`): The input image.
        keep_dim (bool, optional): Whether to keep the number of dimensions of
            the input image. Default: ``False``.

    Returns:
        :obj:`np.ndarray`: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keep_dim:
        out_img = out_img[..., None]
    return out_img


def rgb2gray(img, keep_dim=False):
    """
    Convert an RGB image to a grayscale image.

    Args:
        img (:obj:`np.ndarray`): The input image.
        keep_dim (bool, optional): Whether to keep the number of dimensions of
            the input image. Default: ``False``.

    Returns:
        :obj:`np.ndarray`: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if keep_dim:
        out_img = out_img[..., None]
    return out_img


def gray2bgr(img):
    """
    Convert a grayscale image to a BGR image.

    Args:
        img (:obj:`np.ndarray`): The input image.

    Returns:
        :obj:`np.ndarray`: The converted BGR image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return out_img


def gray2rgb(img):
    """
    Convert a grayscale image to an RGB image.

    Args:
        img (:obj:`np.ndarray`): The input image.

    Returns:
        :obj:`np.ndarray`: The converted RGB image.
    """
    img = img[..., None] if img.ndim == 2 else img
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return out_img


def _convert_color_factory(src, dst):
    code = getattr(cv2, 'COLOR_{}2{}'.format(src, dst))

    src_with_article = '{} {}'.format('a' if src == 'BGR' else 'an', src)
    dst_with_article = '{} {}'.format('a' if dst == 'BGR' else 'an', dst)

    def _convert_color(img):
        return cv2.cvtColor(img, code)

    _convert_color.__doc__ = """
    Convert {} image to {} image.

    Args:
        img (:obj:`np.ndarray`): The input image.

    Returns:
        :obj:`np.ndarray`: The converted {} image.
    """.format(src_with_article, dst_with_article, dst)

    return _convert_color


bgr2rgb = _convert_color_factory('BGR', 'RGB')
rgb2bgr = _convert_color_factory('RGB', 'BGR')
bgr2hsv = _convert_color_factory('BGR', 'HSV')
hsv2bgr = _convert_color_factory('HSV', 'BGR')
bgr2hls = _convert_color_factory('BGR', 'HLS')
hls2bgr = _convert_color_factory('HLS', 'BGR')
