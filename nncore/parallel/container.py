# Copyright (c) Ye Liu. All rights reserved.

import torch

import nncore


@nncore.bind_getter('stack', 'pad_value', 'pad_dims', 'to_gpu')
@nncore.bind_method('data', ['size', 'dim'])
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

        self.data = data
        self._stack = stack
        self._pad_value = pad_value
        self._pad_dims = pad_dims
        self._to_gpu = to_gpu

    def __repr__(self):
        return ('{}(data={}, stack={}, pad_value={}, pad_dims={}, to_gpu={})'.
                format(self.__class__.__name__, self.data, self._stack,
                       self._pad_value, self._pad_dims, self._to_gpu))

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)
