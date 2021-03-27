# Copyright (c) Ye Liu. All rights reserved.

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.parallel._functions import Scatter, _get_stream

from .container import DataContainer


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
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        stream = streams[0] if output.numel() > 0 else None
        with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
            output = output.cuda(devices[0], non_blocking=True)
        return output
    else:
        raise Exception("unknown type '{}'".format(type(input)))


def _synchronize_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                _synchronize_stream(output[i * chunk_size + j], [devices[i]],
                                    [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception("unknown type '{}'".format(type(output)))


def _get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = _get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception("unknown type '{}'".format(type(input)))


def _scatter_forward(target_gpus, input):
    input_device = _get_input_device(input)
    streams = None
    if input_device == -1:
        streams = [_get_stream(device) for device in target_gpus]

    outputs = _scatter_stream(input, target_gpus, streams)
    if streams is not None:
        _synchronize_stream(outputs, target_gpus, streams)

    return tuple(outputs)


def _scatter(inputs, target_gpus, dim=0):

    def _scatter_map(obj):
        if isinstance(obj, DataContainer):
            return _scatter_forward(target_gpus,
                                    obj.data) if obj.to_gpu else obj.data
        elif isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        elif isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(_scatter_map, obj)))
        elif isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(_scatter_map, obj))))
        elif isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(_scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    try:
        return _scatter_map(inputs)
    finally:
        _scatter_map = None


def _scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    inputs = _scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = _scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    return tuple(inputs), tuple(kwargs)


class NNDataParallel(DataParallel):
    """
    A :obj:`nn.DataParallel` class with :obj:`DataContainer` support.
    """

    def scatter(self, inputs, kwargs, device_ids):
        return _scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


class NNDistributedDataParallel(DistributedDataParallel):
    """
    A :obj:`nn.DistributedDataParallel` class with :obj:`DataContainer`
    support.
    """

    def to_kwargs(self, inputs, kwargs, device_id):
        return _scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def scatter(self, inputs, kwargs, device_ids):
        return _scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
