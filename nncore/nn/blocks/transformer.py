# Copyright (c) Ye Liu. All rights reserved.

from math import log

import torch
import torch.nn as nn

import nncore
from ..builder import MODELS, build_act_layer, build_norm_layer
from ..bundle import Parameter, Sequential
from ..init import kaiming_init_, xavier_init_


@MODELS.register()
@nncore.bind_getter('dims', 'learnable', 'p', 'max_len')
class PositionalEncoding(nn.Module):
    """
    Positional Encoding introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        learnable (bool, optional): Whether the positional encoding is
            learnable. Default: ``True``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        max_len (int, optional): The maximum length of the input sequence.
            Default: ``5000``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, dims, learnable=True, p=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self._dims = dims
        self._learnable = learnable
        self._p = p
        self._max_len = max_len

        if learnable:
            self.pe = Parameter(1, max_len, dims)
        else:
            pos = torch.arange(max_len).unsqueeze(1)
            div = (torch.arange(0, dims, 2) * (-log(10000.0) / dims)).exp()
            pe = torch.zeros(1, max_len, dims)
            pe[0, :, 0::2] = (pos * div).sin()
            pe[0, :, 1::2] = (pos * div).cos()
            self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=p)

    def __repr__(self):
        return ('{}(dims={}, learnable={}, p={}, max_len={})'.format(
            self.__class__.__name__, self._dims, self._learnable, self._p,
            self._max_len))

    def forward(self, x):
        pe = self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)
        pe = self.dropout(pe)
        return pe


@MODELS.register()
@nncore.bind_getter('dims', 'q_dims', 'k_dims', 'v_dims', 'h_dims', 'o_dims',
                    'heads', 'p', 'bias')
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention introduced in [1].

    Args:
        dims (int): The input feature dimensions.
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
                 heads=8,
                 p=0.1,
                 bias=True):
        super(MultiHeadAttention, self).__init__()

        self._q_dims = dims
        self._k_dims = k_dims or dims
        self._v_dims = v_dims or dims
        self._h_dims = h_dims or dims
        self._o_dims = o_dims or dims
        self._heads = heads
        self._p = p
        self._bias = bias
        self._head_dims = self._h_dims // heads

        self.q = nn.Linear(self._q_dims, self._h_dims, bias=bias)
        self.k = nn.Linear(self._k_dims, self._h_dims, bias=bias)
        self.v = nn.Linear(self._v_dims, self._h_dims, bias=bias)
        self.m = nn.Linear(self._h_dims, self._o_dims, bias=bias)

        self.drop1 = build_norm_layer('drop', p=p)
        self.drop2 = build_norm_layer('drop', p=p)

        self.reset_parameters()

    def __repr__(self):
        return ('{}(q_dims={}, k_dims={}, v_dims={}, h_dims={}, o_dims={}, '
                'heads={}, p={}, bias={})'.format(self.__class__.__name__,
                                                  self._q_dims, self._k_dims,
                                                  self._v_dims, self._h_dims,
                                                  self._o_dims, self._heads,
                                                  self._p, self._bias))

    def reset_parameters(self):
        for m in (self.q, self.k, self.v, self.m):
            xavier_init_(m)

    def forward(self, q, k=None, v=None, mask=None):
        v = v if torch.is_tensor(v) else k if torch.is_tensor(k) else q
        k = k if torch.is_tensor(k) else q

        q = self.q(q).transpose(0, 1).contiguous()
        k = self.k(k).transpose(0, 1).contiguous()
        v = self.v(v).transpose(0, 1).contiguous()

        b = q.size(1) * self._heads

        q = q.view(-1, b, self._head_dims).transpose(0, 1)
        k = k.view(-1, b, self._head_dims).transpose(0, 1)
        v = v.view(-1, b, self._head_dims).transpose(0, 1)

        att = torch.bmm(q, k.transpose(1, 2)) / self._head_dims**0.5

        if mask is not None:
            mask = torch.where(mask > 0, .0, float('-inf'))
            mask = mask.repeat_interleave(self._heads, dim=0)
            att += mask.unsqueeze(1).expand(-1, att.size(1), -1)

        att = att.softmax(-1)

        if self.drop1 is not None:
            att = self.drop1(att)

        m = torch.bmm(att, v).transpose(0, 1).contiguous()
        m = m.view(m.size(0), -1, self._h_dims).transpose(0, 1)
        m = self.m(m)

        if self.drop2 is not None:
            m = self.drop2(m)

        return m


@MODELS.register()
@nncore.bind_getter('dims', 'ratio', 'p')
class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        ratio (float, optional): The ratio of hidden layer dimensions with
            respect to the input dimensions. Default: ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 ratio=4,
                 p=0.1,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(FeedForwardNetwork, self).__init__()

        self._dims = dims
        self._ratio = ratio
        self._p = p
        self._h_dims = int(dims * ratio)

        self.mapping = Sequential(
            nn.Linear(dims, self._h_dims), build_act_layer(act_cfg),
            build_norm_layer('drop', p=p), nn.Linear(self._h_dims, dims),
            build_norm_layer('drop', p=p))

        self.reset_parameters()

    def __repr__(self):
        return '{}(dims={}, ratio={}, p={})'.format(self.__class__.__name__,
                                                    self._dims, self._ratio,
                                                    self._p)

    def reset_parameters(self):
        for m in self.mapping:
            if isinstance(m, nn.Linear):
                kaiming_init_(m)

    def forward(self, x):
        x = self.mapping(x)
        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'ratio', 'p', 'pre_norm')
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        heads (int, optional): The number of attention heads. Default: ``1``.
        ratio (float, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        pre_norm (bool, optional): Whether to apply the normalization before
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
                 heads=8,
                 ratio=4,
                 p=0.1,
                 pre_norm=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(TransformerEncoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._p = p
        self._pre_norm = pre_norm

        self.att = MultiHeadAttention(dims, heads=heads, p=p)
        self.ffn = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, x, pe=None, mask=None):
        if self._pre_norm:
            v = self.norm1(x)
            q = k = v if pe is None else v + pe
            d = self.att(q, k, v, mask=mask)
            x = x + d

            d = self.norm2(x)
            d = self.ffn(d)
            x = x + d
        else:
            q = k = x if pe is None else x + pe
            d = self.att(q, k, x, mask=mask)
            x = self.norm1(x + d)

            d = self.ffn(x)
            x = self.norm2(x + d)

        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'ratio', 'p', 'pre_norm')
class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        heads (int, optional): The number of attention heads. Default: ``1``.
        ratio (int, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        pre_norm (bool, optional): Whether to apply the normalization before
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
                 heads=8,
                 ratio=4,
                 p=0.1,
                 pre_norm=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(TransformerDecoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._p = p
        self._pre_norm = pre_norm

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.ffn = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, x, mem, q_pe=None, k_pe=None, mask=None):
        if self._pre_norm:
            v = self.norm1(x)
            q = k = v if q_pe is None else v + q_pe
            d = self.att1(q, k, v, mask=mask)
            x = x + d

            q = self.norm2(x)
            q = q if q_pe is None else q + q_pe
            k = mem if k_pe is None else mem + k_pe
            d = self.att2(q, k, mem, mask=mask)
            x = x + d

            d = self.norm3(x)
            d = self.ffn(d)
            x = x + d
        else:
            q = k = x if q_pe is None else x + q_pe
            d = self.att1(q, k, x, mask=mask)
            x = self.norm1(x + d)

            q = x if q_pe is None else x + q_pe
            k = mem if k_pe is None else mem + k_pe
            d = self.att2(q, k, mem, mask=mask)
            x = self.norm2(x + d)

            d = self.ffn(x)
            x = self.norm3(x + d)

        return x
