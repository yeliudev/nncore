# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore
from ..builder import (MODULES, NORMS, build_act_layer, build_conv_layer,
                       build_norm_layer)
from ..init import constant_init_, kaiming_init_


@MODULES.register()
@nncore.bind_getter('in_channels', 'out_channels', 'kernel_size', 'stride',
                    'padding', 'dilation', 'groups', 'bias', 'order',
                    'with_norm', 'with_act')
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
            convolution layer. If ``None``, ``nn.Conv2d`` will be used.
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
        self._order = order
        self._with_norm = 'norm' in order and norm_cfg is not None
        self._with_act = 'act' in order and act_cfg is not None

        if bias != 'auto':
            self._bias = bias
        elif self._with_norm:
            self._bias = norm_cfg['type'] if isinstance(
                norm_cfg, dict) else norm_cfg in NORMS.group('drop')
        else:
            self._bias = True

        self.conv = build_conv_layer(
            conv_cfg or 'Conv2d',
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups)

        if self._with_norm:
            self.norm = build_norm_layer(norm_cfg, dims=out_channels)

        if self._with_act:
            self.act = build_act_layer(act_cfg)

        self.init_weights()

    def init_weights(self):
        kaiming_init_(self.conv)
        if self._with_norm:
            constant_init_(self.norm)

    def forward(self, x):
        for layer in self._order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self._with_norm:
                x = self.norm(x)
            elif layer == 'act' and self._with_act:
                x = self.act(x)
        return x


@MODULES.register()
class DepthwiseSeparableConvModule(nn.Module):
    """
    Depthwise Separable Convolution introduced in [1].

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
        last_norm (bool, optional): Whether to add a normalization layer after
            the pointwise convolution layer. Default: ``True``.
        last_act (bool, optional): Whether to add an activation layer after
            the pointwise convolution layer. Default: ``True``.

    References:
        1. Howard et al. (https://arxiv.org/abs/1704.04861)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 last_norm=True,
                 last_act=True,
                 **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()

        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            **kwargs)

        _kwargs = kwargs.copy()
        order = tuple(
            o for o in _kwargs.get('order', ('conv', 'norm', 'act'))
            if (o == 'conv' or (o == 'norm' and last_norm) or (
                o == 'act' and last_act)))

        self.pointwise_conv = ConvModule(
            in_channels, out_channels, 1, order=order, **_kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


def build_conv_modules(dims,
                       kernel_size,
                       separable=False,
                       last_norm=False,
                       last_act=False,
                       **kwargs):
    """
    Build a sequential module list containing convolution, normalization, and
    activation layers.

    Args:
        dims (list[int]): The sequence of numbers of dimensions of channels.
        kernel_size (list[int] | int): The size or list of sizes of the
            convolution kernel.
        separable (bool, optional): Whether to use depthwise separable
            convolution module. Default: ``False``.
        last_norm (bool, optional): Whether to add a normalization layer after
            the last convolution layer. Default: ``False``.
        last_act (bool, optional): Whether to add an activation layer after
            the last convolution layer. Default: ``False``.

    Returns:
        :obj:`nn.Sequential` or :obj:`ConvModule` or \
            :obj:`DepthwiseSeparableConvModule`: The constructed module.
    """
    Module = DepthwiseSeparableConvModule if separable else ConvModule

    _kwargs = kwargs.copy()
    layers = []

    if isinstance(kernel_size, (int, tuple)):
        kernel_size = [kernel_size] * (len(dims) - 1)

    for i in range(len(dims) - 1):
        if i == len(dims) - 2:
            if separable:
                _kwargs.update(dict(last_norm=last_norm, last_act=last_act))
            else:
                _kwargs['order'] = tuple(
                    o for o in _kwargs.get('order', ('conv', 'norm', 'act'))
                    if (o == 'conv' or (o == 'norm' and last_norm) or (
                        o == 'act' and last_act)))

        module = Module(dims[i], dims[i + 1], kernel_size[i], **_kwargs)
        layers.append(module)

    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]
