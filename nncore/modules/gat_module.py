# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

from .bricks import NORM_LAYERS, GATConv, build_act_layer, build_norm_layer


class GATModule(nn.Module):
    """
    A module that bundles gat-norm-activation layers.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        heads (int, optional): Number of attention heads. Default: ``1``.
        bias (str or bool, optional): Whether to add the bias term in the gat
            layer. If ``bias='auto'``, the module will decide it automatically
            base on whether it has a norm layer. Default: ``'auto'``.
        norm_cfg (dict, optional): The config of norm layer. Default:
            ``dict(type='BN1d')``.
        act_cfg (dict, optional): The config of activation layer. Default:
            ``dict(type='ReLU', inplace=True)``.
        order (tuple[str], optional): The order of gat/norm/activation layers.
            It is expected to be a sequence of ``'msg_pass'``, ``'norm'`` and
            ``'act'``. Default: ``('msg_pass', 'norm', 'act')``.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 heads=1,
                 bias='auto',
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 order=('msg_pass', 'norm', 'act'),
                 **kwargs):
        super(GATModule, self).__init__()
        self.with_norm = 'norm' in order and norm_cfg is not None
        self.with_act = 'act' in order and act_cfg is not None
        self.order = order

        self.msg_pass = GATConv(
            in_features,
            out_features,
            heads,
            bias=bias if bias != 'auto' else not self.with_norm,
            **kwargs)

        if self.with_norm:
            assert norm_cfg['type'] in NORM_LAYERS.group('1d')
            if 'Drop' not in norm_cfg['type']:
                norm_cfg = norm_cfg.copy()
                norm_cfg.setdefault('num_features', out_features * heads)
            self.norm = build_norm_layer(norm_cfg)

        if self.with_act:
            self.act = build_act_layer(act_cfg)

    def forward(self, x, graph):
        """
        Args:
            x (:obj:`torch.Tensor[N, M]`): The input node features.
            graph (:obj:`torch.Tensor[N, N]`): The graph structure where
                ``graph[i, j] == 0`` means there is an link from node ``i`` to
                node ``j`` while ``graph[i, j] == -inf`` means not.
        """
        for layer in self.order:
            if layer == 'msg_pass':
                x = self.msg_pass(x, graph)
            elif layer == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_act:
                x = self.act(x)
        return x


def build_gat_modules(dims, heads=None, with_last_act=False, **kwargs):
    """
    Build a module list containing gat-norm-actvation layers.

    Args:
        dims (list[int]): The sequence of numbers of dimensions of features.
        heads (list[int], optional): The sequence of numbers of attention
            heads in gat layers. If not specified, all the gat layers will have
            only ``1`` head. Default: ``None``.
        with_last_act (bool, optional): Whether to add an activation layer
            after the last gat layer. Default: ``False``.

    Returns:
        :obj:`nn.ModuleList`: The constructed module list.
    """
    assert 'concat' not in kwargs
    _kwargs = kwargs.copy()

    if heads is None:
        heads = [1] * (len(dims) - 1)

    last_out_features, layers = dims[0], []
    for i in range(1, len(dims)):
        if i == len(dims) - 1:
            _kwargs['concat'] = False
            if not with_last_act:
                _kwargs['order'] = ('msg_pass', )

        module = GATModule(
            last_out_features, dims[i], heads=heads[i - 1], **_kwargs)
        last_out_features = dims[i] * heads[i - 1]

        layers.append(module)

    return nn.ModuleList(layers)
