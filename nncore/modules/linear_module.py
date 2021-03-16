# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn
from torch.nn.modules.dropout import _DropoutNd

from .bricks import build_act_layer, build_norm_layer


class LinearModule(nn.Module):

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

        if self.with_norm:
            _norm_cfg = norm_cfg.copy()
            if _norm_cfg['type'] not in ('GN', 'LN'):
                _norm_cfg.setdefault('num_features', out_features)
            self.norm = build_norm_layer(_norm_cfg)

        if self.with_act:
            self.act = build_act_layer(act_cfg)

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias if bias != 'auto' else not self.with_norm
            or isinstance(self.norm, _DropoutNd))

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
    _kwargs = kwargs.copy()
    layers = []

    for i in range(len(dims) - 1):
        if not with_last_act and i == len(dims) - 2:
            _kwargs['order'] = ('linear', )

        module = LinearModule(dims[i], dims[i + 1], **_kwargs)
        layers.append(module)

    return nn.Sequential(*layers)
