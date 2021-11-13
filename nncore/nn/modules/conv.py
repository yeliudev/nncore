# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore
from ..builder import (MODULES, NORMS, build_act_layer, build_conv_layer,
                       build_norm_layer)
from ..init import constant_init_, kaiming_init_


@MODULES.register()
@nncore.bind_getter('in_channels', 'out_channels', 'kernel_size', 'stride',
                    'padding', 'dilation', 'groups', 'bias', 'order')
class ConvModule(nn.Module):
    """
    A module that bundles convolution, normalization, and activation layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple[int] | int): Size of the convolution kernel.
        stride (tuple[int] | int, optional): Stride of the convolution.
            Default: ``1``.
        padding (tuple[int] | int | str, optional): Padding added to the input.
            Default: ``0``.
        dilation (tuple[int] | int, optional): Spacing among neighbouring
            kernel elements. Default: ``1``.
        groups (int, optional): Number of blocked connections from input to
            output channels. Default: ``1``.
        bias (str | bool, optional): Whether to add the bias term in the
            convolution layer. If ``bias='auto'``, the module will decide it
            automatically base on whether it has a normalization layer.
            Default: ``'auto'``.
        conv_cfg (dict | str | None, optional): The config or name of the
            convolution layer. If not specified, ``nn.Conv2d`` will be used.
            Default: ``None``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``None``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
        order (tuple[str], optional): The order of layers. It is expected to
            be a sequence of ``'conv'``, ``'norm'``, and ``'act'``. Default:
            ``('conv', 'norm', 'act')``.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert 'conv' in order

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

        _map = dict(conv=True, norm=norm_cfg, act=act_cfg)
        self._order = tuple(o for o in order if _map[o] is not None)

        if self.with_norm:
            _pos = self._order.index('norm') - self._order.index('conv')

        if bias != 'auto':
            self._bias = bias
        elif self.with_norm:
            _typ = norm_cfg['type'] if isinstance(norm_cfg, dict) else norm_cfg
            self._bias = _typ in NORMS.group('drop') or _pos != 1
        else:
            self._bias = True

        for layer in self._order:
            if layer == 'conv':
                self.conv = build_conv_layer(
                    conv_cfg or 'Conv2d',
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups)
            elif layer == 'norm':
                self.norm = build_norm_layer(
                    norm_cfg, dims=out_channels if _pos > 0 else in_channels)
            else:
                self.act = build_act_layer(act_cfg)

        self.init_weights()

    @property
    def with_norm(self):
        return 'norm' in self._order

    @property
    def with_act(self):
        return 'act' in self._order

    def init_weights(self):
        kaiming_init_(self.conv)
        if self.with_norm:
            constant_init_(self.norm)

    def forward(self, x):
        for layer in self._order:
            x = getattr(self, layer)(x)
        return x


def build_conv_modules(dims,
                       kernels,
                       last_norm=False,
                       last_act=False,
                       default=None,
                       **kwargs):
    """
    Build a sequential module list containing convolution, normalization, and
    activation layers.

    Args:
        dims (list[int]): The sequence of numbers of dimensions of channels.
        kernels (list[int] | int): The size or list of sizes of the
            convolution kernels.
        last_norm (bool, optional): Whether to add a normalization layer after
            the last convolution layer. Default: ``False``.
        last_act (bool, optional): Whether to add an activation layer after
            the last convolution layer. Default: ``False``.
        default (any, optional): The default value when the ``dims`` is not
            valid. Default: ``None``.

    Returns:
        :obj:`nn.Sequential` | :obj:`ConvModule`: The constructed module.
    """
    if not nncore.is_seq_of(dims, int):
        return default

    _kwargs = kwargs.copy()
    _layers = [last_norm or 'norm', last_act or 'act']
    layers = []

    if isinstance(kernels, (int, tuple)):
        kernels = [kernels] * (len(dims) - 1)

    for i in range(len(dims) - 1):
        if i == len(dims) - 2:
            order = list(_kwargs.get('order', ['conv', 'norm', 'act']))
            while order[-1] in _layers:
                order.pop()
            _kwargs['order'] = tuple(order)

        module = ConvModule(dims[i], dims[i + 1], kernels[i], **_kwargs)
        layers.append(module)

    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]
