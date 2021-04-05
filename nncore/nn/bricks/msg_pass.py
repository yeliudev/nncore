# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import nncore

MESSAGE_PASSINGS = nncore.Registry('message passing')


@MESSAGE_PASSINGS.register()
@nncore.bind_getter('in_features', 'out_features')
class GCN(nn.Module):
    """
    Graph Convolutional Layer introduced in [1].

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Whether to add the bias term. Default: ``True``.

    References:
        1. Kipf et al. (https://arxiv.org/abs/1609.02907)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._with_bias = bias

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))

        if self._with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def _compute_norm(self, graph):
        graph = graph.t()

        deg = graph.new_tensor([graph[i].sum() for i in range(graph.size(0))])
        deg_inv_sqrt = deg.pow(-0.5).diag()

        norm = torch.mm(deg_inv_sqrt, graph)
        norm = torch.mm(norm, deg_inv_sqrt)

        return norm

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self._with_bias:
            nn.init.constant_(self.bias, 0)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self._in_features, self._out_features, self._with_bias)

    def forward(self, x, graph):
        """
        Args:
            x (:obj:`torch.Tensor[N, M]`): The input node features.
            graph (:obj:`torch.Tensor[N, N]`): The graph structure where
                ``graph[i, j] == n (n > 0)`` means there is an link with
                weight ``n`` from node ``i`` to node ``j`` while
                ``graph[i, j] == 0`` means not.
        """
        assert x.size(0) == graph.size(0) == graph.size(1)
        n = self._compute_norm(graph)

        h = torch.mm(x, self.weight)
        y = torch.mm(n, h)

        if self._with_bias:
            y += self.bias

        return y


@MESSAGE_PASSINGS.register()
@nncore.bind_getter('in_features', 'out_features', 'k')
class SGC(GCN):
    """
    Simple Graph Convolutional Layer introduced in [1].

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        k (int, optional): Number of layers to be stacked.
        bias (bool, optional): Whether to add the bias term. Default: ``True``.

    References:
        1. Wu et al. (https://arxiv.org/abs/1902.07153)
    """

    def __init__(self, in_features, out_features, k=1, bias=True):
        super(SGC, self).__init__(in_features, out_features, bias=bias)
        self._k = k

    def _compute_norm(self, graph):
        norm = _norm = super(SGC, self)._compute_norm(graph)
        for _ in range(self._k - 1):
            norm = torch.mm(norm, _norm)
        return norm

    def extra_repr(self):
        return 'in_features={}, out_features={}, k={}, bias={}'.format(
            self._in_features, self._out_features, self._k, self._with_bias)


@MESSAGE_PASSINGS.register()
@nncore.bind_getter('in_features', 'out_features', 'heads', 'p',
                    'negative_slope', 'concat', 'residual')
class GAT(nn.Module):
    """
    Graph Attention Layer introduced in [1].

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        heads (int, optional): Number of attention heads. Default: ``1``.
        p (float, optional): The probability of dropping elements. Default:
            ``0``.
        negative_slope (float, optional): The negative slope of
            :obj:`LeakyReLU`. Default: ``0.2``.
        concat (bool, optional): Whether to concatenate the features from
            different attention heads. Default: ``True``.
        residual (bool, optional): Whether to add residual connections.
            Default: ``True``.
        bias (bool, optional): Whether to add the bias term. Default: ``True``.

    References:
        1. Veličković et al. (https://arxiv.org/abs/1710.10903)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 heads=1,
                 p=0,
                 negative_slope=0.2,
                 concat=True,
                 residual=True,
                 bias=True):
        super(GAT, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._heads = heads
        self._p = p
        self._negative_slope = negative_slope
        self._concat = concat
        self._residual = residual
        self._head_features = int(out_features / (heads if concat else 1))
        self._map_residual = in_features != self._head_features
        self._with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(heads, in_features, self._head_features))
        self.weight_i = nn.Parameter(
            torch.Tensor(heads, self._head_features, 1))
        self.weight_j = nn.Parameter(
            torch.Tensor(heads, self._head_features, 1))

        if self._map_residual:
            self.weight_r = nn.Parameter(
                torch.Tensor(in_features, self._head_features * heads))

        if self._with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))

        self.dropout = nn.Dropout(p=p)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        self.reset_parameters()

    def __repr__(self):
        return ('{}(in_features={}, out_features={}, heads={}, p={}, '
                'negative_slope={}, concat={}, residual={}, bias={})'.format(
                    self.__class__.__name__, self._in_features,
                    self._out_features, self._heads, self._p,
                    self._negative_slope, self._concat, self._residual,
                    self._with_bias))

    def reset_parameters(self):
        for weight in (self.weight, self.weight_i, self.weight_j):
            nn.init.xavier_uniform_(weight)
        if self._map_residual:
            nn.init.xavier_uniform_(self.weight_r)
        if self._with_bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, graph):
        """
        Args:
            x (:obj:`torch.Tensor[N, M]`): The input node features.
            graph (:obj:`torch.Tensor[N, N]`): The graph structure where
                ``graph[i, j] == n (n > 0)`` means there is a link from node
                ``i`` to node ``j`` while ``graph[i, j] == 0`` means not.
        """
        assert x.size(0) == graph.size(0) == graph.size(1)

        x = self.dropout(x)
        h = torch.matmul(x[None, :], self.weight)

        coe_i = torch.bmm(h, self.weight_i)
        coe_j = torch.bmm(h, self.weight_j).transpose(1, 2)
        coe = self.leaky_relu(coe_i + coe_j)

        graph = torch.where(graph > 0, .0, float('-inf')).t()
        att = self.dropout(F.softmax(coe + graph, dim=-1))

        y = torch.bmm(att, h).transpose(0, 1).contiguous()

        if self._residual:
            if self._map_residual:
                y += torch.mm(x, self.weight_r).view(-1, self._heads,
                                                     self._head_features)
            else:
                y += x[:, None]

        if self._concat:
            y = y.view(-1, self._out_features)
        else:
            y = y.mean(dim=1)

        if self._with_bias:
            y += self.bias

        return y


def build_msg_pass_layer(cfg, **kwargs):
    """
    Build a message passing layer from a dict. This method searches for layers
    in :obj:`MESSAGE_PASSINGS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict or str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return nncore.build_object(cfg, [MESSAGE_PASSINGS, nn], **kwargs)
