# Copyright (c) Ye Liu. All rights reserved.

import torch

import nncore


@nncore.bind_getter('stack', 'pad_value', 'pad_dims', 'cpu_only')
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
        cpu_only (bool, optional): Whether to keep the data on CPU only
            Default: ``False``.
    """

    def __init__(self,
                 data,
                 stack=True,
                 pad_value=0,
                 pad_dims=-1,
                 cpu_only=False):
        assert pad_dims in (None, -1, 1, 2, 3)

        if cpu_only:
            stack = False
            pad_dims = None

        self._data = data
        self._stack = stack
        self._pad_value = pad_value
        self._pad_dims = pad_dims if pad_dims != -1 else data.dim()
        self._cpu_only = cpu_only

    def __repr__(self):
        return ('{}(data={}, stack={}, pad_value={}, pad_dims={}, cpu_only={})'
                .format(self.__class__.__name__, self._data, self._stack,
                        self._pad_value, self._pad_dims, self._cpu_only))

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        if torch.is_tensor(self._data):
            return self._data.type()
        else:
            return type(self._data)
