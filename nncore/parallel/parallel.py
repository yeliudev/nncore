# Copyright (c) Ye Liu. All rights reserved.

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.parallel._functions import Function, Scatter, _get_stream

from .container import DataContainer


class _Scatter(Function):

    @staticmethod
    def forward(target_gpus, input):
        input_device = _get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            streams = [_get_stream(device) for device in target_gpus]

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
        if torch.is_tensor(obj):
            if target_gpus != [-1]:
                return Scatter.apply(target_gpus, None, dim, obj)
            else:
                return _Scatter.forward(target_gpus, obj)
        elif isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return _Scatter.forward(target_gpus, obj.data)
        elif isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(_scatter_map, obj)))
        elif isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(_scatter_map, obj))))
        elif isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(_scatter_map, obj.items()))))
        return [obj for _ in target_gpus]

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
    A :obj:`nn.DataParallel` class with :obj:`DataContainer` support. This
    class only bundles single-device modules.
    """

    def __init__(self, module, device_ids=None, dim=0, **kwargs):
        assert device_ids is None or len(device_ids) <= 1
        super(NNDataParallel, self).__init__(
            module,
            device_ids=[0] if device_ids is None else device_ids,
            **kwargs)
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
    """

    def __init__(self,
                 module,
                 device_ids=None,
                 broadcast_buffers=False,
                 **kwargs):
        assert device_ids is None or len(device_ids) <= 1

        if device_ids is None:
            if torch.cuda.is_available():
                device_ids = [torch.cuda.current_device()]
                module = module.cuda()
        elif len(device_ids) == 1:
            module = module.to('cuda:{}'.format(device_ids[0]))

        super(NNDistributedDataParallel, self).__init__(
            module,
            device_ids=device_ids,
            broadcast_buffers=broadcast_buffers,
            **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return _scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def to_kwargs(self, inputs, kwargs, device_id):
        return _scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def forward(self, *inputs, **kwargs):
        if self.device_ids:
            return super(NNDistributedDataParallel,
                         self).forward(*inputs, **kwargs)
        else:
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
