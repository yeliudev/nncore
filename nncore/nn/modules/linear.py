# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore
from ..builder import MODULES, NORMS, build_act_layer, build_norm_layer
from ..init import constant_init_, kaiming_init_


@MODULES.register()
@nncore.bind_getter('in_features', 'out_features', 'bias', 'order')
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

        self._in_features = in_features
        self._out_features = out_features

        _map = dict(linear=True, norm=norm_cfg, act=act_cfg)
        self._order = tuple(o for o in order if _map[o] is not None)

        if self.with_norm:
            _pos = self._order.index('norm') - self._order.index('linear')

        if bias != 'auto':
            self._bias = bias
        elif self.with_norm:
            _typ = norm_cfg['type'] if isinstance(norm_cfg, dict) else norm_cfg
            self._bias = _typ in NORMS.group('drop') or _pos != 1
        else:
            self._bias = True

        for layer in self._order:
            if layer == 'linear':
                self.linear = nn.Linear(
                    in_features, out_features, bias=self._bias)
            elif layer == 'norm':
                self.norm = build_norm_layer(
                    norm_cfg, dims=out_features if _pos > 0 else in_features)
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
        kaiming_init_(self.linear)
        if self.with_norm:
            constant_init_(self.norm)

    def forward(self, x):
        for layer in self._order:
            x = getattr(self, layer)(x)
        return x


def build_linear_modules(dims,
                         last_norm=False,
                         last_act=False,
                         default=None,
                         **kwargs):
    """
    Build a multi-layer perceptron (MLP).

    Args:
        dims (list[int]): The sequence of numbers of dimensions of features.
        last_norm (bool, optional): Whether to add a normalization layer after
            the last linear layer. Default: ``False``.
        last_act (bool, optional): Whether to add an activation layer after
            the last linear layer. Default: ``False``.
        default (any, optional): The default value when the ``dims`` is not
            valid. Default: ``None``.

    Returns:
        :obj:`nn.Sequential` | :obj:`LinearModule`: The constructed module.
    """
    if not nncore.is_seq_of(dims, int):
        return default

    _kwargs = kwargs.copy()
    _layers = [last_norm or 'norm', last_act or 'act']
    layers = []

    for i in range(len(dims) - 1):
        if i == len(dims) - 2:
            order = list(_kwargs.get('order', ['linear', 'norm', 'act']))
            while order[-1] in _layers:
                order.pop()
            _kwargs['order'] = tuple(order)

        module = LinearModule(dims[i], dims[i + 1], **_kwargs)
        layers.append(module)

    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]
