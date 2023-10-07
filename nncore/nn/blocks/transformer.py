# Copyright (c) Ye Liu. Licensed under the MIT License.

from math import log

import torch
import torch.nn as nn

import nncore
from ..builder import MODELS, build_act_layer, build_norm_layer
from ..bundle import Parameter, Sequential
from ..init import init_module_
from .norm import DropPath


@MODELS.register()
@nncore.bind_getter('dims', 'learnable', 'max_len')
class PositionalEncoding(nn.Module):
    """
    Positional Encoding introduced in [1].

    Args:
        dims (int): Input feature dimensions.
        learnable (bool, optional): Whether the positional encoding is
            learnable. Default: ``True``.
        dropout (float, optional): Dropout probability. Default: ``0.0``.
        max_len (int, optional): Maximum length of the input sequence. Default:
            ``5000``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, dims, learnable=True, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self._dims = dims
        self._learnable = learnable
        self._dropout = dropout
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

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def __repr__(self):
        return ('{}(dims={}, learnable={}, dropout={}, max_len={})'.format(
            self.__class__.__name__, self._dims, self._learnable,
            self._dropout, self._max_len))

    def forward(self, x):
        pe = self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)
        if self._dropout > 0:
            pe = self.dropout(pe)
        return pe


@MODELS.register()
@nncore.bind_getter('dims', 'q_dims', 'k_dims', 'v_dims', 'h_dims', 'o_dims',
                    'heads', 'bias', 'init_cfg')
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention introduced in [1].

    Args:
        dims (int): Input feature dimensions.
        k_dims (int | None, optional): Dimensions of key matrix. If not
            specified, it will be the same as ``q_dims``. Default: ``None``.
        v_dims (int | None, optional): Dimensions of value matrix. If not
            specified, it will be the same as ``q_dims``. Default: ``None``.
        h_dims (int | None, optional): Hidden dimensions. If not specified, it
            will be the same as ``q_dims``. Default: ``None``.
        o_dims (int | None, optional): Output dimensions. If not specified, it
            will be the same as ``q_dims``. Default: ``None``.
        heads (int, optional): Number of attention heads. Default: ``8``.
        att_dropout (float, optional): Dropout probability for the attention
            map. Default: ``0.0``.
        out_dropout (float, optional): Dropout probability for outputs.
            Default: ``0.0``.
        bias (bool, optional): Whether to add the bias term. Default: ``True``.
        init_cfg (dict | str, optional): The config for module initialization.
            Default: ``dict(type='xavier')``.

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
                 att_dropout=0.0,
                 out_dropout=0.0,
                 bias=True,
                 init_cfg=dict(type='xavier')):
        super(MultiHeadAttention, self).__init__()

        self._q_dims = dims
        self._k_dims = k_dims or dims
        self._v_dims = v_dims or dims
        self._h_dims = h_dims or dims
        self._o_dims = o_dims or dims
        self._heads = heads
        self._att_dropout = att_dropout
        self._out_dropout = out_dropout
        self._bias = bias
        self._head_dims = self._h_dims // heads
        self._init_cfg = init_cfg

        self.q = nn.Linear(self._q_dims, self._h_dims, bias=bias)
        self.k = nn.Linear(self._k_dims, self._h_dims, bias=bias)
        self.v = nn.Linear(self._v_dims, self._h_dims, bias=bias)
        self.m = nn.Linear(self._h_dims, self._o_dims, bias=bias)

        if att_dropout > 0:
            self.att_dropout = nn.Dropout(p=att_dropout)

        if out_dropout > 0:
            self.out_dropout = nn.Dropout(p=out_dropout)

        self.reset_parameters()

    def __repr__(self):
        return ('{}(q_dims={}, k_dims={}, v_dims={}, h_dims={}, o_dims={}, '
                'heads={}, att_dropout={}, out_dropout={}, bias={})'.format(
                    self.__class__.__name__, self._q_dims, self._k_dims,
                    self._v_dims, self._h_dims, self._o_dims, self._heads,
                    self._att_dropout, self._out_dropout, self._bias))

    def reset_parameters(self):
        for m in (self.q, self.k, self.v, self.m):
            init_module_(m, self._init_cfg)

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
            att += mask.unsqueeze(1)

        att = att.softmax(-1)

        if self._att_dropout > 0:
            att = self.att_dropout(att)

        m = torch.bmm(att, v).transpose(0, 1).contiguous()
        m = m.view(m.size(0), -1, self._h_dims).transpose(0, 1)
        m = self.m(m)

        if self._out_dropout > 0:
            m = self.out_dropout(m)

        return m


@MODELS.register()
@nncore.bind_getter('dims', 'ratio', 'ffn_dropout', 'out_dropout', 'init_cfg')
class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network introduced in [1].

    Args:
        dims (int): Input feature dimensions.
        ratio (float, optional): The ratio of hidden layer dimensions with
            respect to the input dimensions. Default: ``4``.
        ffn_dropout (float, optional): Dropout probability for hidden layers.
            Default: ``0.0``.
        out_dropout (float, optional): Dropout probability for outputs.
            Default: ``0.0``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
        init_cfg (dict | str, optional): The config for module initialization.
            Default: ``dict(type='xavier')``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 ratio=4,
                 ffn_dropout=0.0,
                 out_dropout=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=dict(type='xavier')):
        super(FeedForwardNetwork, self).__init__()

        self._dims = dims
        self._ratio = ratio
        self._ffn_dropout = ffn_dropout
        self._out_dropout = out_dropout
        self._h_dims = int(dims * ratio)
        self._init_cfg = init_cfg

        self.mapping = Sequential()
        self.mapping.append(nn.Linear(dims, self._h_dims))
        self.mapping.append(build_act_layer(act_cfg))

        if ffn_dropout > 0:
            self.mapping.append(nn.Dropout(p=ffn_dropout))

        self.mapping.append(nn.Linear(self._h_dims, dims))

        if out_dropout > 0:
            self.mapping.append(nn.Dropout(p=out_dropout))

        self.reset_parameters()

    def __repr__(self):
        return '{}(dims={}, ratio={}, ffn_dropout={}, out_dropout={})'.format(
            self.__class__.__name__, self._dims, self._ratio,
            self._ffn_dropout, self._out_dropout)

    def reset_parameters(self):
        for m in self.mapping:
            if isinstance(m, nn.Linear):
                init_module_(m, self._init_cfg)

    def forward(self, x):
        x = self.mapping(x)
        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'ratio', 'att_dropout', 'ffn_dropout',
                    'att_out_dropout', 'ffn_out_dropout', 'pre_norm', 'bias',
                    'init_cfg')
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        heads (int, optional): The number of attention heads. Default: ``8``.
        ratio (float, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``4``.
        att_dropout (float, optional): Dropout probability for the attention
            map. Default: ``0.0``.
        ffn_dropout (float, optional): Dropout probability for hidden layers
            of feed forward network. Default: ``0.0``.
        att_out_dropout (float, optional): Dropout probability for the outputs
            of attention block. Default: ``0.0``.
        ffn_out_dropout (float, optional): Dropout probability for the outputs
            of feed forward network. Default: ``0.0``.
        droppath (float, optional): Probability of dropping paths. Default:
            ``0.1``.
        pre_norm (bool, optional): Whether to apply the normalization before
            instead of after each layer. Default: ``True``.
        bias (bool, optional): Whether to add the bias term in the attention
            block. Default: ``True``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``dict(type='LN')``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
        init_cfg (dict | str, optional): The config for module initialization.
            Default: ``dict(type='xavier')``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 att_dropout=0.0,
                 ffn_dropout=0.0,
                 att_out_dropout=0.0,
                 ffn_out_dropout=0.0,
                 droppath=0.1,
                 pre_norm=True,
                 bias=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=dict(type='xavier')):
        super(TransformerEncoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._att_dropout = att_dropout
        self._ffn_dropout = ffn_dropout
        self._att_out_dropout = att_out_dropout
        self._ffn_out_dropout = ffn_out_dropout
        self._droppath = droppath
        self._pre_norm = pre_norm
        self._bias = bias
        self._init_cfg = init_cfg

        self.att = MultiHeadAttention(
            dims,
            heads=heads,
            att_dropout=att_dropout,
            out_dropout=att_out_dropout,
            bias=bias,
            init_cfg=init_cfg)
        self.ffn = FeedForwardNetwork(
            dims,
            ratio=ratio,
            ffn_dropout=ffn_dropout,
            out_dropout=ffn_out_dropout,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)

        if droppath > 0:
            self.droppath = DropPath(p=droppath)

    def forward(self, x, pe=None, mask=None):
        if self._pre_norm:
            v = self.norm1(x)
            q = k = v if pe is None else v + pe
            d = self.att(q, k, v, mask=mask)
            if self._droppath > 0:
                d = self.droppath(d)
            x = x + d

            d = self.norm2(x)
            d = self.ffn(d)
            if self._droppath > 0:
                d = self.droppath(d)
            x = x + d
        else:
            q = k = x if pe is None else x + pe
            d = self.att(q, k, x, mask=mask)
            if self._droppath > 0:
                d = self.droppath(d)
            x = self.norm1(x + d)

            d = self.ffn(x)
            if self._droppath > 0:
                d = self.droppath(d)
            x = self.norm2(x + d)

        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'ratio', 'att_dropout', 'ffn_dropout',
                    'att_out_dropout', 'ffn_out_dropout', 'pre_norm', 'bias',
                    'init_cfg')
class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        heads (int, optional): The number of attention heads. Default: ``8``.
        ratio (int, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``4``.
        att_dropout (float, optional): Dropout probability for the attention
            maps. Default: ``0.0``.
        ffn_dropout (float, optional): Dropout probability for hidden layers
            of feed forward network. Default: ``0.0``.
        att_out_dropout (float, optional): Dropout probability for the outputs
            of attention blocks. Default: ``0.0``.
        ffn_out_dropout (float, optional): Dropout probability for the outputs
            of feed forward network. Default: ``0.0``.
        droppath (float, optional): Probability of dropping paths. Default:
            ``0.1``.
        pre_norm (bool, optional): Whether to apply the normalization before
            instead of after each layer. Default: ``True``.
        bias (bool, optional): Whether to add the bias term in the attention
            block. Default: ``True``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``dict(type='LN')``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
        init_cfg (dict | str, optional): The config for module initialization.
            Default: ``dict(type='xavier')``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 att_dropout=0.0,
                 ffn_dropout=0.0,
                 att_out_dropout=0.0,
                 ffn_out_dropout=0.0,
                 droppath=0.1,
                 pre_norm=True,
                 bias=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=dict(type='xavier')):
        super(TransformerDecoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._att_dropout = att_dropout
        self._ffn_dropout = ffn_dropout
        self._att_out_dropout = att_out_dropout
        self._ffn_out_dropout = ffn_out_dropout
        self._droppath = droppath
        self._pre_norm = pre_norm
        self._bias = bias
        self._init_cfg = init_cfg

        self.att1 = MultiHeadAttention(
            dims,
            heads=heads,
            att_dropout=att_dropout,
            out_dropout=att_out_dropout,
            bias=bias,
            init_cfg=init_cfg)
        self.att2 = MultiHeadAttention(
            dims,
            heads=heads,
            att_dropout=att_dropout,
            out_dropout=att_out_dropout,
            bias=bias,
            init_cfg=init_cfg)
        self.ffn = FeedForwardNetwork(
            dims,
            ratio=ratio,
            ffn_dropout=ffn_dropout,
            out_dropout=ffn_out_dropout,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)

        if droppath > 0:
            self.droppath = DropPath(p=droppath)

    def forward(self, x, mem, q_pe=None, k_pe=None, q_mask=None, k_mask=None):
        if self._pre_norm:
            v = self.norm1(x)
            q = k = v if q_pe is None else v + q_pe
            d = self.att1(q, k, v, mask=q_mask)
            if self._droppath > 0:
                d = self.droppath(d)
            x = x + d

            q = self.norm2(x)
            q = q if q_pe is None else q + q_pe
            k = mem if k_pe is None else mem + k_pe
            d = self.att2(q, k, mem, mask=k_mask)
            if self._droppath > 0:
                d = self.droppath(d)
            x = x + d

            d = self.norm3(x)
            d = self.ffn(d)
            if self._droppath > 0:
                d = self.droppath(d)
            x = x + d
        else:
            q = k = x if q_pe is None else x + q_pe
            d = self.att1(q, k, x, mask=q_mask)
            if self._droppath > 0:
                d = self.droppath(d)
            x = self.norm1(x + d)

            q = x if q_pe is None else x + q_pe
            k = mem if k_pe is None else mem + k_pe
            d = self.att2(q, k, mem, mask=k_mask)
            if self._droppath > 0:
                d = self.droppath(d)
            x = self.norm2(x + d)

            d = self.ffn(x)
            if self._droppath > 0:
                d = self.droppath(d)
            x = self.norm3(x + d)

        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'ratio', 'att_dropout', 'ffn_dropout',
                    'att_out_dropout', 'ffn_out_dropout', 'pre_norm', 'bias',
                    'init_cfg')
class CrossAttentionLayer(nn.Module):
    """
    Cross Attention Layer.

    Args:
        dims (int): The input feature dimensions.
        heads (int, optional): The number of attention heads. Default: ``8``.
        ratio (int, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``4``.
        att_dropout (float, optional): Dropout probability for the attention
            maps. Default: ``0.0``.
        ffn_dropout (float, optional): Dropout probability for hidden layers
            of feed forward network. Default: ``0.0``.
        att_out_dropout (float, optional): Dropout probability for the outputs
            of attention blocks. Default: ``0.0``.
        ffn_out_dropout (float, optional): Dropout probability for the outputs
            of feed forward network. Default: ``0.0``.
        droppath (float, optional): Probability of dropping paths. Default:
            ``0.1``.
        pre_norm (bool, optional): Whether to apply the normalization before
            instead of after each layer. Default: ``True``.
        bias (bool, optional): Whether to add the bias term in the attention
            block. Default: ``True``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``dict(type='LN')``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.
        init_cfg (dict | str, optional): The config for module initialization.
            Default: ``dict(type='xavier')``.
    """

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 att_dropout=0.0,
                 ffn_dropout=0.0,
                 att_out_dropout=0.0,
                 ffn_out_dropout=0.0,
                 droppath=0.1,
                 pre_norm=True,
                 bias=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=dict(type='xavier')):
        super(CrossAttentionLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._att_dropout = att_dropout
        self._ffn_dropout = ffn_dropout
        self._att_out_dropout = att_out_dropout
        self._ffn_out_dropout = ffn_out_dropout
        self._droppath = droppath
        self._pre_norm = pre_norm
        self._bias = bias
        self._init_cfg = init_cfg

        self.att1 = MultiHeadAttention(
            dims,
            heads=heads,
            att_dropout=att_dropout,
            out_dropout=att_out_dropout,
            bias=bias,
            init_cfg=init_cfg)
        self.att2 = MultiHeadAttention(
            dims,
            heads=heads,
            att_dropout=att_dropout,
            out_dropout=att_out_dropout,
            bias=bias,
            init_cfg=init_cfg)
        self.ffn1 = FeedForwardNetwork(
            dims,
            ratio=ratio,
            ffn_dropout=ffn_dropout,
            out_dropout=ffn_out_dropout,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.ffn2 = FeedForwardNetwork(
            dims,
            ratio=ratio,
            ffn_dropout=ffn_dropout,
            out_dropout=ffn_out_dropout,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)
        self.norm4 = build_norm_layer(norm_cfg, dims=dims)

        if droppath > 0:
            self.droppath = DropPath(p=droppath)

    def forward(self, a, b, a_mask=None, b_mask=None):
        _a, _b = a, b

        if self._pre_norm:
            q = self.norm1(a)
            d = self.att1(q, _b, _b, mask=b_mask)
            if self._droppath > 0:
                d = self.droppath(d)
            a = a + d

            q = self.norm2(b)
            d = self.att2(q, _a, _a, mask=a_mask)
            if self._droppath > 0:
                d = self.droppath(d)
            b = b + d

            d = self.norm3(a)
            d = self.ffn1(d)
            if self._droppath > 0:
                d = self.droppath(d)
            a = a + d

            d = self.norm4(b)
            d = self.ffn2(d)
            if self._droppath > 0:
                d = self.droppath(d)
            b = b + d
        else:
            d = self.att1(a, _b, _b, mask=b_mask)
            if self._droppath > 0:
                d = self.droppath(d)
            a = self.norm1(a + d)

            d = self.att2(b, _a, _a, mask=a_mask)
            if self._droppath > 0:
                d = self.droppath(d)
            b = self.norm2(b + d)

            d = self.ffn1(a)
            if self._droppath > 0:
                d = self.droppath(d)
            a = self.norm3(a + d)

            d = self.ffn2(b)
            if self._droppath > 0:
                d = self.droppath(d)
            b = self.norm4(b + d)

        return a, b
