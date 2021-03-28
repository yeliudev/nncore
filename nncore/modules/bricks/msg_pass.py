# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import nncore

MASSAGE_PASSING_LAYERS = nncore.Registry('massage passing layer')


@MASSAGE_PASSING_LAYERS.register()
class GATConv(nn.Module):
    """
    Graph Attention Layer introduced in [1].

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        heads (int, optional): Number of attention heads. Default: ``1``.
        dropout (float, optional): The probability of dropping elements.
            Default: ``0``.
        negative_slope (float, optional): The negative slope of ``LeakyReLU``.
            Default: ``0.2``.
        concat (bool, optional): Whether to concatenate the features from
            different attention heads. Default: ``True``.
        residual (bool, optional): Whether to add residual connections.
            Default: ``True``.
        bias (bool, optional): Whether to add the bias term. Default: ``True``.

    References:
        1. Lin et al. (https://arxiv.org/abs/1710.10903)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 heads=1,
                 dropout=0,
                 negative_slope=0.2,
                 concat=True,
                 residual=True,
                 bias=True):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self.residual = residual

        self.w = nn.Parameter(torch.Tensor(heads, in_features, out_features))
        self.h_i = nn.Parameter(torch.Tensor(heads, out_features, 1))
        self.h_j = nn.Parameter(torch.Tensor(heads, out_features, 1))

        if residual:
            self.r = nn.Parameter(
                torch.Tensor(in_features, out_features * heads))

        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(out_features * (heads if concat else 1)))
        else:
            self.register_parameter('bias', None)

        self.d = nn.Dropout(p=dropout)
        self.a = nn.LeakyReLU(negative_slope=negative_slope)

        self.reset_parameters()

    def __repr__(self):
        attrs = [
            '{}={}'.format(attr, getattr(self, attr)) for attr in [
                'in_features', 'out_features', 'heads', 'dropout', 'concat',
                'residual'
            ]
        ] + ['bias={}'.format(self.bias is not None)]
        return '{}({})'.format(self.__class__.__name__, ', '.join(attrs))

    def reset_parameters(self):
        for w in (self.w, self.h_i, self.h_j):
            nn.init.xavier_uniform_(w)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, graph):
        """
        Args:
            x (:obj:`torch.Tensor[N, M]`): The input node features.
            graph (:obj:`torch.Tensor[N, N]`): The graph structure where
                ``graph[i, j] == 0`` means there is an link from node ``i`` to
                node ``j`` while ``graph[i, j] == -inf`` means not.
        """
        x = self.d(x)
        h = torch.matmul(x[None, :], self.w)

        h_i = torch.bmm(h, self.h_i)
        h_j = torch.bmm(h, self.h_j)

        att = self.a(h_i + h_j.transpose(1, 2))
        att = self.d(F.softmax(att + graph, dim=-1))

        y = torch.bmm(att, h).transpose(0, 1).contiguous()

        if self.residual:
            if y.size(-1) == x.size(-1):
                y += x[:, None]
            else:
                y += torch.matmul(x, self.r).view(-1, self.heads,
                                                  self.out_features)

        if self.concat:
            y = y.view(-1, self.out_features * self.heads)
        else:
            y = y.mean(dim=1)

        if self.bias is not None:
            y += self.bias

        return y


def build_msg_layer(cfg, **kwargs):
    return nncore.build_object(cfg, [MASSAGE_PASSING_LAYERS, nn], **kwargs)
