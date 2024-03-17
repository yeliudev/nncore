# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.scatter_gather import _is_namedtuple

import nncore
from .container import DataContainer


def _scatter(inputs, target_gpus, dim=0):

    def _scatter_map(obj):
        if torch.is_tensor(obj) and target_gpus != [-1]:
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, DataContainer):
            return _scatter_map(
                obj.data, [-1] if obj.cpu_only else target_gpus, dim=dim)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(_scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(_scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(_scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(_scatter_map, obj.items()))]
        return [obj for _ in target_gpus]

    try:
        res = _scatter_map(inputs)
    finally:
        _scatter_map = None

    return res


def _scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    inputs = _scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = _scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend(() for _ in range(len(kwargs) - len(inputs)))
    elif len(kwargs) < len(inputs):
        kwargs.extend({} for _ in range(len(inputs) - len(kwargs)))
    return tuple(inputs), tuple(kwargs)


class NNDataParallel(DataParallel):
    """
    A :obj:`nn.DataParallel` class with :obj:`DataContainer` support. This
    class only bundles single-device modules.

    Args:
        module (:obj:`nn.Module`): The module to be bundled.
        device_id (int, optional): The device id to be used. ``-1`` means CPU.
            Default: ``0``.
    """

    def __init__(self, module, device_id=0, **kwargs):
        assert isinstance(device_id, int)
        assert 'device_ids' not in kwargs and 'output_device' not in kwargs

        if device_id == -1:
            logger = nncore.get_logger()
            logger.warn('{} is running on CPU'.format(self.__class__.__name__))

        device_ids = [device_id] if device_id >= 0 else []

        super(NNDataParallel, self).__init__(
            module, device_ids=device_ids, output_device=device_ids, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return _scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        if self.device_ids:
            return super(NNDataParallel, self).forward(*inputs, **kwargs)
        else:
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])


class NNDistributedDataParallel(DistributedDataParallel):
    """
    A :obj:`nn.DistributedDataParallel` class with :obj:`DataContainer`
    support. This class only bundles single-device modules.

    Args:
        module (:obj:`nn.Module`): The module to be bundled.
        device_id (int, optional): The device id to be used. ``-1`` means CPU.
            Default: ``0``.
    """

    def __init__(self, module, device_id=0, **kwargs):
        assert isinstance(device_id, int) and 'device_ids' not in kwargs

        if device_id >= 0:
            module = module.to('cuda:{}'.format(device_id))
            device_ids = [device_id]
        else:
            logger = nncore.get_logger()
            logger.warn('{} is running on CPU'.format(self.__class__.__name__))
            device_ids = None

        super(NNDistributedDataParallel, self).__init__(
            module, device_ids=device_ids, **kwargs)

    def to_kwargs(self, inputs, kwargs, device_id):
        return _scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def forward(self, *inputs, **kwargs):
        if self.device_ids:
            return super(NNDistributedDataParallel,
                         self).forward(*inputs, **kwargs)
        else:
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
