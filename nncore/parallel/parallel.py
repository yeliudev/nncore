# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.parallel._functions import Scatter, _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple

import nncore
from .container import DataContainer


class _Scatter(torch.autograd.Function):

    @staticmethod
    def forward(target_gpus, input):
        input_device = _get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            streams = [
                _get_stream(torch.device('cuda', gpu_id))
                for gpu_id in target_gpus
            ]
        outputs = _scatter_stream(input, target_gpus, streams)
        if streams is not None:
            _sync_stream(outputs, target_gpus, streams)
        return tuple(outputs)


def _get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = _get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif torch.is_tensor(input):
        return input.get_device() if input.is_cuda else -1
    else:
        raise TypeError('unknown type {}'.format(type(input)))


def _scatter_stream(input, devices, streams=None):
    if streams is None:
        streams = [None] * len(devices)
    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            _scatter_stream(input[i], [devices[i // chunk_size]],
                            [streams[i // chunk_size]])
            for i in range(len(input))
        ]
        return outputs
    elif torch.is_tensor(input):
        output = input.contiguous()
        stream = streams[0] if output.numel() > 0 else None
        if devices != [-1]:
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=True)
        return output
    else:
        raise TypeError('unknown type {}'.format(type(input)))


def _sync_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                _sync_stream(output[i * chunk_size + j], [devices[i]],
                             [streams[i]])
    elif torch.is_tensor(output):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise TypeError('unknown type {}'.format(type(output)))


def _scatter(inputs, target_gpus, dim=0):

    def _scatter_map(obj):
        if torch.is_tensor(obj) and target_gpus != [-1]:
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, DataContainer):
            return obj.data if obj.cpu_only else _Scatter.forward(
                target_gpus, obj.data)
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


def _get_device(device_id=None):
    if device_id is not None:
        return device_id
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return -1


class NNDataParallel(DataParallel):
    """
    A :obj:`nn.DataParallel` class with :obj:`DataContainer` support. This
    class only bundles single-device modules.

    Args:
        module (:obj:`nn.Module`): The module to be bundled.
        device_id (int | None, optional): The device id to be used. ``None``
            means using the default device, and ``-1`` means CPU. Default:
            ``None``.
    """

    def __init__(self, module, device_id=None, dim=0, **kwargs):
        assert isinstance(device_id, int) or device_id is None
        assert 'device_ids' not in kwargs and 'output_device' not in kwargs

        device_id = _get_device(device_id)

        if device_id >= 0:
            super(NNDataParallel, self).__init__(
                module,
                device_ids=[device_id],
                output_device=device_id,
                **kwargs)
        else:
            logger = nncore.get_logger()
            logger.warn('{} is running on CPU'.format(self.__class__.__name__))

            super(DataParallel, self).__init__()
            torch._C._log_api_usage_once('torch.nn.parallel.DataParallel')

            self.module = module
            self.device_ids = []
            self.dim = dim

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
        device_id (int | None, optional): The device id to be used. ``None``
            means using the default device, and ``-1`` means CPU. Default:
            ``None``.
    """

    def __init__(self, module, device_id=None, **kwargs):
        assert isinstance(device_id, int) or device_id is None
        assert 'device_ids' not in kwargs and 'output_device' not in kwargs

        device_id = _get_device(device_id)

        if device_id >= 0:
            module = module.to('cuda:{}'.format(device_id))
            device_ids = [device_id]
        else:
            logger = nncore.get_logger()
            logger.warn('{} is running on CPU'.format(self.__class__.__name__))
            device_ids = None

        super(NNDistributedDataParallel, self).__init__(
            module, device_ids=device_ids, **kwargs)

    def _pre_forward(self, *inputs, **kwargs):
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            inputs, kwargs = inputs[0], kwargs[0]
        return super(NNDistributedDataParallel,
                     self)._pre_forward(*inputs, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return _scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        if self.device_ids:
            return super(NNDistributedDataParallel,
                         self).forward(*inputs, **kwargs)
        else:
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
