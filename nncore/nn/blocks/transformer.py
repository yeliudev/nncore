# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn

import nncore
from ..builder import MODELS, build_act_layer, build_norm_layer
from ..init import kaiming_init_
from .bundle import Sequential


@MODELS.register()
@nncore.bind_getter('dims', 'k_dims', 'v_dims', 'h_dims', 'o_dims', 'heads',
                    'p', 'bias')
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention introduced in [1].

    Args:
        dims (int): The dimensions of query matrix.
        k_dims (int | None, optional): The dimensions of key matrix. If not
            specified, it will be the same as ``q_dims``. Default: ``None``.
        v_dims (int | None, optional): The dimensions of value matrix. If not
            specified, it will be the same as ``q_dims``. Default: ``None``.
        h_dims (int | None, optional): The hidden dimensions. If not specified,
            it will be the same as ``q_dims``. Default: ``None``.
        o_dims (int | None, optional): The output dimensions. If not specified,
            it will be the same as ``q_dims``. Default: ``None``.
        heads (int, optional): The number of attention heads. Default: ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        bias (bool, optional): Whether to add the bias term. Default: ``True``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 k_dims=None,
                 v_dims=None,
                 h_dims=None,
                 o_dims=None,
                 heads=1,
                 p=0.1,
                 bias=True):
        super(MultiHeadAttention, self).__init__()

        self._dims = dims
        self._k_dims = k_dims or dims
        self._v_dims = v_dims or dims
        self._h_dims = h_dims or dims
        self._o_dims = o_dims or dims
        self._heads = heads
        self._p = p
        self._bias = bias
        self._head_dims = self._h_dims // heads

        self.proj_q = nn.Linear(self._dims, self._h_dims, bias=bias)
        self.proj_k = nn.Linear(self._k_dims, self._h_dims, bias=bias)
        self.proj_v = nn.Linear(self._v_dims, self._h_dims, bias=bias)
        self.proj_m = nn.Linear(self._h_dims, self._o_dims, bias=bias)

        self.dropout = build_norm_layer('Drop', p=p, inplace=True)
        self.reset_parameters()

    def __repr__(self):
        return ('{}(dims={}, k_dims={}, v_dims={}, h_dims={}, o_dims={}, '
                'heads={}, p={}, bias={})'.format(self.__class__.__name__,
                                                  self._dims, self._k_dims,
                                                  self._v_dims, self._h_dims,
                                                  self._o_dims, self._heads,
                                                  self._p, self._bias))

    def reset_parameters(self):
        for m in (self.proj_q, self.proj_k, self.proj_v, self.proj_m):
            kaiming_init_(m)

    def forward(self, q, k=None, v=None, mask=None):
        v = v if torch.is_tensor(v) else k if torch.is_tensor(k) else q
        k = k if torch.is_tensor(k) else q

        q = self.proj_q(q).view(-1, q.size(1), self._head_dims)
        k = self.proj_k(k).view(-1, k.size(1), self._head_dims)
        v = self.proj_v(v).view(-1, v.size(1), self._head_dims)

        att = torch.bmm(q, k.transpose(1, 2)) / self._h_dims**0.5

        if mask is not None:
            mask = torch.where(mask > 0, .0, float('-inf'))
            att += mask.repeat_interleave(self._heads, dim=0)

        att = att.softmax(-1)

        if self.dropout is not None:
            att = self.dropout(att)

        m = torch.bmm(att, v).view(-1, att.size(1), self._h_dims)
        m = self.proj_m(m)

        return m


@MODELS.register()
@nncore.bind_getter('dims', 'ratio', 'p')
class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network introduced in [1].

    Args:
        dims (int): The input dimensions.
        ratio (int, optional): The ratio of hidden layer dimensions. Default:
            ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 ratio=1,
                 p=0.1,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(FeedForwardNetwork, self).__init__()

        self._dims = dims
        self._ratio = ratio
        self._p = p
        self._h_dims = dims * ratio

        self.mapping = Sequential(
            nn.Linear(dims, self._h_dims), build_act_layer(act_cfg),
            build_norm_layer('Drop', p=p, inplace=True),
            nn.Linear(self._h_dims, dims))

    def __repr__(self):
        return '{}(dims={}, ratio={}, p={})'.format(self.__class__.__name__,
                                                    self._dims, self._ratio,
                                                    self._p)

    def forward(self, x):
        x = self.mapping(x)
        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'p', 'ratio', 'norm_first')
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer introduced in [1].

    Args:
        dims (int): The input dimensions.
        heads (int, optional): The number of attention heads. Default: ``1``.
        ratio (int, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        norm_first (bool, optional): Whether to apply the normalization before
            instead of after each layer. Default: ``True``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``dict(type='LN')``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 heads=1,
                 ratio=1,
                 p=0.1,
                 norm_first=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(TransformerEncoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._p = p
        self._norm_first = norm_first

        self.att = MultiHeadAttention(dims, heads=heads, p=p)
        self.ffn = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)

        self.drop1 = build_norm_layer('Drop', p=p, inplace=True)
        self.drop2 = build_norm_layer('Drop', p=p, inplace=True)

    def forward(self, x, mask=None):
        if self._norm_first:
            d = self.norm1(x)
            d = self.att(d, mask=mask)
            x = x + self.drop1(d)

            d = self.norm2(x)
            d = self.ffn(d)
            x = x + self.drop2(d)
        else:
            d = self.att(x, mask=mask)
            d = self.drop1(d)
            x = self.norm1(d + x)

            d = self.ffn(x)
            d = self.drop2(d)
            x = self.norm2(d + x)

        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'p', 'ratio', 'norm_first')
class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer introduced in [1].

    Args:
        dims (int): The input dimensions.
        heads (int, optional): The number of attention heads. Default: ``1``.
        ratio (int, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        norm_first (bool, optional): Whether to apply the normalization before
            instead of after each layer. Default: ``True``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``dict(type='LN')``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 heads=1,
                 ratio=1,
                 p=0.1,
                 norm_first=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(TransformerDecoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._p = p
        self._norm_first = norm_first

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.ffn = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)

        self.drop1 = build_norm_layer('Drop', p=p, inplace=True)
        self.drop2 = build_norm_layer('Drop', p=p, inplace=True)
        self.drop3 = build_norm_layer('Drop', p=p, inplace=True)

    def forward(self, x, m, mask=None):
        if self._norm_first:
            d = self.norm1(x)
            d = self.att1(d, mask=mask)
            x = x + self.drop1(d)

            d = self.norm2(x)
            d = self.att2(d, m, mask=mask)
            x = x + self.drop2(d)

            d = self.norm3(x)
            d = self.ffn(d)
            x = x + self.drop3(d)
        else:
            d = self.att1(x, mask=mask)
            d = self.drop1(d)
            x = self.norm1(d + x)

            d = self.att2(x, m, mask=mask)
            d = self.drop2(d)
            x = self.norm2(d + x)

            d = self.ffn(x)
            d = self.drop3(d)
            x = self.norm3(d + x)

        return x
