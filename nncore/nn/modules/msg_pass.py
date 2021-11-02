# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore
from ..builder import (MODULES, NORMS, build_act_layer, build_msg_pass_layer,
                       build_norm_layer)
from ..init import constant_init_


@MODULES.register()
@nncore.bind_getter('in_features', 'out_features', 'bias', 'order',
                    'with_norm', 'with_act')
class MsgPassModule(nn.Module):
    """
    A module that bundles message passing, normalization, and activation
    layers.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (str | bool, optional): Whether to add the bias term in the
            message passing layer. If ``bias='auto'``, the module will decide
            it automatically base on whether it has a normalization layer.
            Default: ``'auto'``.
        msg_pass_cfg (dict | str, optional): The config or name of the message
            passing layer. Default: ``'GCN'``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``None``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
        order (tuple[str], optional): The order of layers. It is expected to
            be a sequence of ``'msg_pass'``, ``'norm'``, and ``'act'``.
            Default: ``('msg_pass', 'norm', 'act')``.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias='auto',
                 msg_pass_cfg='GCN',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 order=('msg_pass', 'norm', 'act'),
                 **kwargs):
        super(MsgPassModule, self).__init__()
        assert 'msg_pass' in order

        self._in_features = in_features
        self._out_features = out_features
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

        self.msg_pass = build_msg_pass_layer(
            msg_pass_cfg,
            in_features=in_features,
            out_features=out_features,
            bias=self._bias,
            **kwargs)

        if self._with_norm:
            self.norm = build_norm_layer(norm_cfg, dims=out_features)

        if self._with_act:
            self.act = build_act_layer(act_cfg)

        self.init_weights()

    def init_weights(self):
        if self._with_norm:
            constant_init_(self.norm)

    def forward(self, x, graph):
        """
        Args:
            x (:obj:`torch.Tensor[N, M]`): The input node features.
            graph (:obj:`torch.Tensor[N, N]`): The graph structure where
                ``graph[i, j] == n (n > 0)`` means there is a link with weight
                ``n`` from node ``i`` to node ``j`` while ``graph[i, j] == 0``
                means not.
        """
        for layer in self._order:
            if layer == 'msg_pass':
                x = self.msg_pass(x, graph)
            elif layer == 'norm' and self._with_norm:
                x = self.norm(x)
            elif layer == 'act' and self._with_act:
                x = self.act(x)
        return x


def build_msg_pass_modules(dims, last_norm=False, last_act=False, **kwargs):
    """
    Build a module list containing message passing, normalization, and
    activation layers.

    Args:
        dims (list[int]): The sequence of numbers of dimensions of features.
        last_norm (bool, optional): Whether to add a normalization layer after
            the last message passing layer. Default: ``False``.
        last_act (bool, optional): Whether to add an activation layer after
            the last message passing layer. Default: ``False``.

    Returns:
        :obj:`nn.ModuleList`: The constructed module list.
    """
    _kwargs = kwargs.copy()
    cfg, layers = [], []

    for key, value in _kwargs.items():
        if isinstance(value, list):
            assert len(value) == len(dims) - 1
            cfg.append(key)

    cfg = {k: _kwargs.pop(k) for k in cfg}

    for i in range(len(dims) - 1):
        if i == len(dims) - 2:
            _kwargs['order'] = tuple(
                o for o in _kwargs.get('order', ('linear', 'norm', 'act'))
                if (o == 'linear' or (o == 'norm' and last_norm) or (
                    o == 'act' and last_act)))

            _cfg = _kwargs.get('msg_pass_cfg')
            if _cfg is not None and _cfg['type'] == 'GAT':
                _kwargs['concat'] = False

        _kwargs.update({k: v[i] for k, v in cfg.items()})
        module = MsgPassModule(dims[i], dims[i + 1], **_kwargs)

        layers.append(module)

    return nn.ModuleList(layers)
