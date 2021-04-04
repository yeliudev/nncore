# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore
from ..bricks import NORMS, build_act_layer, build_norm_layer


@nncore.bind_getter('in_features', 'out_features', 'bias', 'order')
class LinearModule(nn.Module):
    """
    A module that bundles linear, normalization and activation layers.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (str or bool, optional): Whether to add the bias term in the
            linear layer. If ``bias='auto'``, the module will decide it
            automatically base on whether it has a normalization layer.
            Default: ``'auto'``.
        norm_cfg (dict, optional): The config of the normalization layer.
            Default: ``dict(type='BN1d')``.
        act_cfg (dict, optional): The config of the activation layer. Default:
            ``dict(type='ReLU', inplace=True)``.
        order (tuple[str], optional): The order of layers. It is expected to
            be a sequence of ``'linear'``, ``'norm'`` and ``'act'``. Default:
            ``('linear', 'norm', 'act')``.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias='auto',
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 order=('linear', 'norm', 'act')):
        super(LinearModule, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias if bias != 'auto' else norm_cfg is None or norm_cfg[
            'type'] in NORMS.group('drop') or 'norm' not in order

        _order = []
        for layer in order:
            if layer == 'linear':
                self.linear = nn.Linear(
                    in_features, out_features, bias=self._bias)
            elif layer == 'norm' and norm_cfg is not None:
                assert norm_cfg['type'] in NORMS.group('1d')
                if 'Drop' not in norm_cfg['type']:
                    _norm_cfg = norm_cfg.copy()
                    _norm_cfg.setdefault('num_features', out_features)
                self.norm = build_norm_layer(_norm_cfg)
            elif layer == 'act' and act_cfg is not None:
                self.act = build_act_layer(act_cfg)
            else:
                raise KeyError(
                    "layer types in order must be 'linear', 'norm' or 'act', "
                    "but got '{}'".format(layer))
            _order.append(layer)

        self._order = tuple(_order)

    def forward(self, x):
        for layer in self._order:
            if layer == 'linear':
                x = self.linear(x)
            elif layer == 'norm':
                x = self.norm(x)
            else:
                x = self.act(x)
        return x


def build_mlp(dims, with_last_act=False, **kwargs):
    """
    Build a multi-layer perceptron (MLP).

    Args:
        dims (list[int]): The sequence of numbers of dimensions of features.
        with_last_act (bool, optional): Whether to add an activation layer
            after the last linear layer. Default: ``False``.

    Returns:
        :obj:`nn.Sequential` The constructed MLP module.
    """
    _kwargs = kwargs.copy()
    layers = []

    for i in range(len(dims) - 1):
        if not with_last_act and i == len(dims) - 2:
            _kwargs['order'] = ('linear', )

        module = LinearModule(dims[i], dims[i + 1], **_kwargs)
        layers.append(module)

    return nn.Sequential(*layers)
