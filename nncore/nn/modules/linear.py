# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore
from ..builder import MODULES, NORMS, build_act_layer, build_norm_layer


@MODULES.register()
@nncore.bind_getter('in_features', 'out_features', 'bias', 'order', 'has_norm',
                    'has_act')
class LinearModule(nn.Module):
    """
    A module that bundles linear, normalization, and activation layers.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (str | bool, optional): Whether to add the bias term in the
            linear layer. If ``bias='auto'``, the module will decide it
            automatically base on whether it has a normalization layer.
            Default: ``'auto'``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``None``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
        order (tuple[str], optional): The order of layers. It is expected to
            be a sequence of ``'linear'``, ``'norm'``, and ``'act'``. Default:
            ``('linear', 'norm', 'act')``.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias='auto',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 order=('linear', 'norm', 'act')):
        super(LinearModule, self).__init__()
        assert 'linear' in order

        self._in_features = in_features
        self._out_features = out_features
        self._order = order
        self._has_norm = 'norm' in order and norm_cfg is not None
        self._has_act = 'act' in order and act_cfg is not None

        if self._has_norm:
            norm_type = norm_cfg['type'] if isinstance(norm_cfg,
                                                       dict) else norm_cfg
            assert norm_type in NORMS.group('1d')

        if bias != 'auto':
            self._bias = bias
        elif self._has_norm:
            self._bias = norm_type in NORMS.group('drop')
        else:
            self._bias = True

        self.linear = nn.Linear(in_features, out_features, bias=self._bias)

        if self._has_norm:
            self.norm = build_norm_layer(norm_cfg, dims=out_features)

        if self._has_act:
            self.act = build_act_layer(act_cfg)

    def forward(self, x):
        for layer in self._order:
            if layer == 'linear':
                x = self.linear(x)
            elif layer == 'norm' and self._has_norm:
                x = self.norm(x)
            elif layer == 'act' and self._has_act:
                x = self.act(x)
        return x


def build_linear_modules(dims,
                         last_norm=False,
                         last_act=False,
                         order=('linear', 'norm', 'act'),
                         **kwargs):
    """
    Build a multi-layer perceptron (MLP).

    Args:
        dims (list[int]): The sequence of numbers of dimensions of features.
        last_norm (bool, optional): Whether to add a normalization layer after
            the last linear layer. Default: ``False``.
        last_act (bool, optional): Whether to add an activation layer after
            the last linear layer. Default: ``False``.
        order (tuple[str], optional): The order of layers. It is expected to
            be a sequence of ``'linear'``, ``'norm'``, and ``'act'``. Default:
            ``('linear', 'norm', 'act')``.

    Returns:
        :obj:`nn.Sequential` or :obj:`LinearModule`: The constructed module.
    """
    _kwargs = kwargs.copy()
    layers = []

    for i in range(len(dims) - 1):
        if i == len(dims) - 2:
            order = list(order)
            if not last_norm and 'norm' in order:
                order.remove('norm')
            if not last_act and 'act' in order:
                order.remove('act')
            order = tuple(order)

        module = LinearModule(dims[i], dims[i + 1], order=order, **_kwargs)
        layers.append(module)

    if len(layers) > 1:
        module = nn.Sequential(*layers)
    else:
        module = layers[0]

    return module
