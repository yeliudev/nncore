# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

from .bricks import NORM_LAYERS, build_act_layer, build_norm_layer


class LinearModule(nn.Module):
    """
    A module that bundles linear-norm-activation layers.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (str or bool, optional): Whether to add the bias term in the
            linear layer. If ``bias='auto'``, the module will decide it
            automatically base on whether it has a norm layer. Default:
            ``'auto'``.
        norm_cfg (dict, optional): The config of norm layer. Default:
            ``dict(type='BN1d')``.
        act_cfg (dict, optional): The config of activation layer. Default:
            ``dict(type='ReLU', inplace=True)``.
        order (tuple[str], optional): The order of linear/norm/activation
            layers. It is expected to be a sequence of ``'linear'``, ``'norm'``
            and ``'act'``. Default: ``('linear', 'norm', 'act')``.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias='auto',
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 order=('linear', 'norm', 'act')):
        super(LinearModule, self).__init__()
        self.with_norm = 'norm' in order and norm_cfg is not None
        self.with_act = 'act' in order and act_cfg is not None
        self.order = order

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias if bias != 'auto' else not self.with_norm)

        if self.with_norm:
            assert norm_cfg['type'] in NORM_LAYERS.group('1d')
            if 'Drop' not in norm_cfg['type']:
                norm_cfg = norm_cfg.copy()
                norm_cfg.setdefault('num_features', out_features)
            self.norm = build_norm_layer(norm_cfg)

        if self.with_act:
            self.act = build_act_layer(act_cfg)

    def forward(self, x):
        for layer in self.order:
            if layer == 'linear':
                x = self.linear(x)
            elif layer == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_act:
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
