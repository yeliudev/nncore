# Copyright (c) Ye Liu. All rights reserved.

from functools import wraps

import torch

import nncore


def _assert_tensor_type(func):

    @wraps(func)
    def _wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.format(
                args[0].__class__.__name__, func.__name__, args[0].datatype))
        return func(*args, **kwargs)

    return _wrapper


@nncore.bind_getter('data', 'stack', 'pad_value', 'pad_dims', 'to_gpu')
class DataContainer(object):
    """
    A wrapper for data to make it easily be padded and be scattered to
    different GPUs.

    Args:
        data (any): The object to be wrapped.
        stack (bool, optional): Whether to stack the data during scattering.
            This argument is valid only when the data is a :obj:`torch.Tensor`.
            Default: ``False``.
        pad_value (int, optional): The value to use for padding. Default:
            ``0``.
        pad_dims (int, optional): Number of dimensions to be padded. Default:
            ``2``.
        to_gpu (bool, optional): Whether to move the data to GPU before
            feeding it to the model. Default: ``True``.
    """

    def __init__(self,
                 data,
                 stack=False,
                 pad_value=0,
                 pad_dims=2,
                 to_gpu=True):
        if pad_dims is not None and data.dim() <= pad_dims:
            raise ValueError('too many dimensions to be padded')

        self._data = data
        self._stack = stack
        self._pad_value = pad_value
        self._pad_dims = pad_dims
        self._to_gpu = to_gpu

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._data)

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @_assert_tensor_type
    def size(self, *args, **kwargs):
        return self._data.size(*args, **kwargs)

    @_assert_tensor_type
    def dim(self):
        return self._data.dim()
