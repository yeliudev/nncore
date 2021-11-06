# Copyright (c) Ye Liu. All rights reserved.

import torch

import nncore


@nncore.bind_getter('stack', 'pad_value', 'pad_dims', 'to_gpu')
@nncore.bind_method('_data', ['size', 'dim'])
class DataContainer(object):
    """
    A wrapper for data to make it easily be padded and be scattered to
    different GPUs.

    Args:
        data (any): The object to be wrapped.
        stack (bool, optional): Whether to stack the data during scattering.
            This argument is valid only when the data is a :obj:`torch.Tensor`.
            Default: ``True``.
        pad_value (int, optional): The padding value. Default: ``0``.
        pad_dims (int, optional): Number of dimensions to be padded. Expected
            values include ``None``, ``-1``, ``1``, ``2``, and ``3``. Default:
            ``-1``.
        to_gpu (bool, optional): Whether to move the data to GPU before
            feeding it to the model. Default: ``True``.
    """

    def __init__(self,
                 data,
                 stack=True,
                 pad_value=0,
                 pad_dims=-1,
                 to_gpu=True):
        assert pad_dims in (None, -1, 1, 2, 3)

        self._data = data
        self._stack = stack
        self._pad_value = pad_value
        self._pad_dims = pad_dims if pad_dims != -1 else data.dim()
        self._to_gpu = to_gpu

    def __repr__(self):
        return ('{}(data={}, stack={}, pad_value={}, pad_dims={}, to_gpu={})'.
                format(self.__class__.__name__, self._data, self._stack,
                       self._pad_value, self._pad_dims, self._to_gpu))

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        if torch.is_tensor(self._data):
            return self._data.type()
        else:
            return type(self._data)
