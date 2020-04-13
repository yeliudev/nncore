# Copyright (c) Ye Liu. All rights reserved.

from functools import wraps

import torch

import nncore


def assert_tensor_type(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.format(
                args[0].__class__.__name__, func.__name__, args[0].datatype))
        return func(*args, **kwargs)

    return wrapper


@nncore.bind_getter('data', 'stack', 'pad_value', 'pad_dims', 'to_gpu')
class DataContainer(object):
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data,
                 stack=False,
                 pad_value=0,
                 pad_dims=2,
                 to_gpu=True):
        if stack:
            if not isinstance(data, torch.Tensor):
                raise TypeError('only tensor type can be stacked')
            elif pad_dims is not None and data.dim() <= pad_dims:
                raise ValueError('too many dimensions to be padded')
            elif not to_gpu:
                raise AttributeError('cpu data can not be stacked')

        self._data = data
        self._stack = stack
        self._pad_value = pad_value
        self._pad_dims = pad_dims
        self._to_gpu = to_gpu

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self._data))

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self._data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self._data.dim()
