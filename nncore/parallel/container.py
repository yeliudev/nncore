# Copyright (c) Ye Liu. All rights reserved.

from functools import wraps

import torch

import nncore


def _assert_tensor_type(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.format(
                args[0].__class__.__name__, func.__name__, args[0].datatype))
        return func(*args, **kwargs)

    return wrapper


@nncore.bind_getter('data', 'stack', 'pad_value', 'pad_dims', 'to_gpu')
class DataContainer(object):

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
