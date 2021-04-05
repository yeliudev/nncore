# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore
from ..bricks import (NORMS, build_act_layer, build_msg_pass_layer,
                      build_norm_layer)


@nncore.bind_getter('in_features', 'out_features', 'bias', 'order')
class MsgPassModule(nn.Module):
    """
    A module that bundles message passing, normalization and activation layers.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (str or bool, optional): Whether to add the bias term in the
            message passing layer. If ``bias='auto'``, the module will decide
            it automatically base on whether it has a normalization layer.
            Default: ``'auto'``.
        msg_pass_cfg (dict, optional): The config of the message passing layer.
            Default: ``dict(type='GCN')``.
        norm_cfg (dict, optional): The config of the normalization layer.
            Default: ``dict(type='BN1d')``.
        act_cfg (dict, optional): The config of the activation layer. Default:
            ``dict(type='ReLU', inplace=True)``.
        order (tuple[str], optional): The order of layers. It is expected to
            be a sequence of ``'msg_pass'``, ``'norm'`` and ``'act'``. Default:
            ``('msg_pass', 'norm', 'act')``.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias='auto',
                 msg_pass_cfg=dict(type='GCN'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 order=('msg_pass', 'norm', 'act'),
                 **kwargs):
        super(MsgPassModule, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias if bias != 'auto' else norm_cfg is None or norm_cfg[
            'type'] in NORMS.group('drop') or 'norm' not in order

        _order = []
        for layer in order:
            if layer == 'msg_pass':
                self.msg_pass = build_msg_pass_layer(
                    msg_pass_cfg,
                    in_features=in_features,
                    out_features=out_features,
                    bias=self._bias,
                    **kwargs)
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
                    "layer types in order must be 'msg_pass', 'norm' or "
                    "'act', but got '{}'".format(layer))
            _order.append(layer)

        self._order = tuple(_order)

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
            elif layer == 'norm':
                x = self.norm(x)
            else:
                x = self.act(x)
        return x


def build_msg_pass_modules(dims, with_last_act=False, **kwargs):
    """
    Build a module list containing message passing, normalization and
    activation layers.

    Args:
        dims (list[int]): The sequence of numbers of dimensions of features.
        with_last_act (bool, optional): Whether to add an activation layer
            after the last message passing layer. Default: ``False``.

    Returns:
        :obj:`nn.ModuleList`: The constructed module list.
    """
    cfg, layers = [], []
    _kwargs = kwargs.copy()

    for key, value in _kwargs.items():
        if isinstance(value, list):
            assert len(value) == len(dims) - 1
            cfg.append(key)
    cfg = {k: _kwargs.pop(k) for k in cfg}

    for i in range(len(dims) - 1):
        if i == len(dims) - 2:
            _cfg = _kwargs.get('msg_pass_cfg')
            if _cfg is not None and _cfg['type'] == 'GAT':
                _kwargs['concat'] = False
            if not with_last_act:
                _kwargs['order'] = ('msg_pass', )

        _kwargs.update({k: v[i] for k, v in cfg.items()})
        module = MsgPassModule(dims[i], dims[i + 1], **_kwargs)

        layers.append(module)

    return nn.ModuleList(layers)
