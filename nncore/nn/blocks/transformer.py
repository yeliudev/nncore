# Copyright (c) Ye Liu. Licensed under the MIT License.

import math

import torch
import torch.nn as nn

import nncore
from ..builder import MODELS, build_act_layer, build_norm_layer
from ..bundle import Sequential
from ..init import init_module_
from .norm import DropPath


@MODELS.register()
@nncore.bind_getter('dims', 'temperature', 'normalize', 'scale', 'offset',
                    'learnable', 'max_len')
class PositionalEncoding(nn.Module):
    """
    Positional Encoding introduced in [1].

    Args:
        dims (int): Input feature dimensions.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Default: ``10000``.
        normalize (bool, optional): Whether to normalize the positional
            encoding. Default: ``False``.
        scale (float, optional): Scale factor for the position encoding.
            It will be used only when ``normalize=True`` Default: ``2 * pi``.
        offset (float): Offset value for normalization. Default: ``0``.
        learnable (bool, optional): Whether the positional encoding is
            learnable. Default: ``True``.
        dropout (float, optional): Dropout probability. Default: ``0.0``.
        max_len (int, optional): Maximum length of the input sequence. Default:
            ``1000``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 offset=0.0,
                 learnable=False,
                 dropout=0.0,
                 max_len=1000):
        super(PositionalEncoding, self).__init__()

        self._dims = dims
        self._temperature = temperature
        self._normalize = normalize
        self._scale = scale
        self._offset = offset
        self._learnable = learnable
        self._dropout = dropout
        self._max_len = max_len

        if learnable:
            self.pe = nn.Embedding(max_len, dims)

            if dropout > 0:
                self.dropout = nn.Dropout(p=dropout)

    def __repr__(self):
        if self._learnable:
            repr_str = ('{}(dims={}, learnable={}, dropout={}, '
                        'max_len={})'.format(self.__class__.__name__,
                                             self._dims, self._learnable,
                                             self._dropout, self._max_len))
        else:
            repr_str = ('{}(dims={}, temperature={}, normalize={}, '
                        'scale={}, offset={})'.format(self.__class__.__name__,
                                                      self._dims,
                                                      self._temperature,
                                                      self._normalize,
                                                      self._scale,
                                                      self._offset))
        return repr_str

    def forward(self, x, mask=None):
        if mask is not None:
            assert x.size(1) == mask.size(1)
            pe = mask.cumsum(1)
        else:
            pe = torch.arange(1, x.size(1) + 1, device=x.device)
            pe = pe.unsqueeze(0).repeat(x.size(0), 1)

        if self._learnable:
            pe = self.pe(pe - 1)

            if self._dropout > 0:
                pe = self.dropout(pe)
        else:
            if self._normalize:
                pe = (pe + self._offset) / (pe[:, -1:] + 1e-6) * self._scale

            dt = torch.arange(self._dims, device=x.device)
            dt = self.temperature**(dt // 2 * 2 / self._dims)

            pe = pe.unsqueeze(-1) / dt
            pe = pe[:, :, 0::2].sin(), pe[:, :, 1::2].cos()
            pe = torch.stack(pe, dim=3).flatten(start_dim=2)

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
        init_cfg (dict | str | None, optional): The initialization config for
            qkv projection layers. Default:
            ``dict(type='xavier', distribution='uniform')``.

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
                 init_cfg=dict(type='xavier', distribution='uniform')):
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
            att += mask if mask.dim() == 3 else mask.unsqueeze(1)

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
        init_cfg (dict | str | None, optional): The initialization config for
            linear layers. Default: ``dict(type='kaiming')``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 ratio=4,
                 ffn_dropout=0.0,
                 out_dropout=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=dict(type='kaiming')):
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
                    'order', 'att_init_cfg', 'ffn_init_cfg')
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
        order (tuple[str], optional): The order of sub-modules. This argument
            should be a sequence of `'self_att'` and `'ffn'`.
            Default: `('self_att', 'ffn')`.
        att_init_cfg (dict | str | None, optional): The initialization config
            for qkv projection layers in the attention block. Default:
            ``dict(type='xavier', distribution='uniform')``.
        ffn_init_cfg (dict | str | None, optional): The initialization config
            for linear layers in the feed forward network. Default:
            ``dict(type='kaiming')``.

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
                 order=('self_att', 'ffn'),
                 att_init_cfg=dict(type='xavier', distribution='uniform'),
                 ffn_init_cfg=dict(type='kaiming')):
        super(TransformerEncoderLayer, self).__init__()

        assert all(o in ('self_att', 'ffn') for o in order)
        assert len(order) == len(set(order))

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
        self._order = order
        self._att_init_cfg = att_init_cfg
        self._ffn_init_cfg = ffn_init_cfg

        for name in order:
            if name == 'self_att':
                self.self_att = MultiHeadAttention(
                    dims,
                    heads=heads,
                    att_dropout=att_dropout,
                    out_dropout=att_out_dropout,
                    bias=bias,
                    init_cfg=att_init_cfg)
                self.self_att_norm = build_norm_layer(norm_cfg, dims=dims)
            else:
                self.ffn = FeedForwardNetwork(
                    dims,
                    ratio=ratio,
                    ffn_dropout=ffn_dropout,
                    out_dropout=ffn_out_dropout,
                    act_cfg=act_cfg,
                    init_cfg=ffn_init_cfg)
                self.ffn_norm = build_norm_layer(norm_cfg, dims=dims)

        if droppath > 0:
            self.droppath = DropPath(p=droppath)

    def forward(self, x, pe=None, mask=None):
        if self._pre_norm:
            for name in self._order:
                if name == 'self_att':
                    v = self.self_att_norm(x)
                    q = k = v if pe is None else v + pe
                    d = self.self_att(q, k, v, mask=mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = x + d
                else:
                    d = self.ffn_norm(x)
                    d = self.ffn(d)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = x + d
        else:
            for name in self._order:
                if name == 'self_att':
                    q = k = x if pe is None else x + pe
                    d = self.self_att(q, k, x, mask=mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = self.self_att_norm(x + d)
                else:
                    d = self.ffn(x)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = self.ffn_norm(x + d)

        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'ratio', 'att_dropout', 'ffn_dropout',
                    'att_out_dropout', 'ffn_out_dropout', 'pre_norm', 'bias',
                    'order', 'att_init_cfg', 'ffn_init_cfg')
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
        order (tuple[str], optional): The order of sub-modules. This argument
            should be a sequence of `'self_att'`, `'cross_att'`, and `'ffn'`.
            Default: `('self_att', 'cross_att', 'ffn')`.
        att_init_cfg (dict | str | None, optional): The initialization config
            for qkv projection layers in the attention block. Default:
            ``dict(type='xavier', distribution='uniform')``.
        ffn_init_cfg (dict | str | None, optional): The initialization config
            for linear layers in the feed forward network. Default:
            ``dict(type='kaiming')``.

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
                 order=('self_att', 'cross_att', 'ffn'),
                 att_init_cfg=dict(type='xavier', distribution='uniform'),
                 ffn_init_cfg=dict(type='kaiming')):
        super(TransformerDecoderLayer, self).__init__()

        assert all(o in ('self_att', 'cross_att', 'ffn') for o in order)
        assert len(order) == len(set(order))

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
        self._order = order
        self._att_init_cfg = att_init_cfg
        self._ffn_init_cfg = ffn_init_cfg

        for name in order:
            if name == 'self_att':
                self.self_att = MultiHeadAttention(
                    dims,
                    heads=heads,
                    att_dropout=att_dropout,
                    out_dropout=att_out_dropout,
                    bias=bias,
                    init_cfg=att_init_cfg)
                self.self_att_norm = build_norm_layer(norm_cfg, dims=dims)
            elif name == 'cross_att':
                self.cross_att = MultiHeadAttention(
                    dims,
                    heads=heads,
                    att_dropout=att_dropout,
                    out_dropout=att_out_dropout,
                    bias=bias,
                    init_cfg=att_init_cfg)
                self.cross_att_norm = build_norm_layer(norm_cfg, dims=dims)
            else:
                self.ffn = FeedForwardNetwork(
                    dims,
                    ratio=ratio,
                    ffn_dropout=ffn_dropout,
                    out_dropout=ffn_out_dropout,
                    act_cfg=act_cfg,
                    init_cfg=ffn_init_cfg)
                self.ffn_norm = build_norm_layer(norm_cfg, dims=dims)

        if droppath > 0:
            self.droppath = DropPath(p=droppath)

    def forward(self, x, mem, q_pe=None, k_pe=None, q_mask=None, k_mask=None):
        if self._pre_norm:
            for name in self._order:
                if name == 'self_att':
                    v = self.self_att_norm(x)
                    q = k = v if q_pe is None else v + q_pe
                    d = self.self_att(q, k, v, mask=q_mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = x + d
                elif name == 'cross_att':
                    q = self.cross_att_norm(x)
                    q = q if q_pe is None else q + q_pe
                    k = mem if k_pe is None else mem + k_pe
                    d = self.cross_att(q, k, mem, mask=k_mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = x + d
                else:
                    d = self.ffn_norm(x)
                    d = self.ffn(d)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = x + d
        else:
            for name in self._order:
                if name == 'self_att':
                    q = k = x if q_pe is None else x + q_pe
                    d = self.self_att(q, k, x, mask=q_mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = self.self_att_norm(x + d)
                elif name == 'cross_att':
                    q = x if q_pe is None else x + q_pe
                    k = mem if k_pe is None else mem + k_pe
                    d = self.cross_att(q, k, mem, mask=k_mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = self.cross_att_norm(x + d)
                else:
                    d = self.ffn(x)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    x = self.ffn_norm(x + d)

        return x


@MODELS.register()
@nncore.bind_getter('dims', 'heads', 'ratio', 'att_dropout', 'ffn_dropout',
                    'att_out_dropout', 'ffn_out_dropout', 'pre_norm', 'bias',
                    'order', 'att_init_cfg', 'ffn_init_cfg')
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
        order (tuple[str], optional): The order of sub-modules. This argument
            should be a sequence of `'cross_att'` and `'ffn'`.
            Default: `('cross_att', 'ffn')`.
        att_init_cfg (dict | str | None, optional): The initialization config
            for qkv projection layers in the attention block. Default:
            ``dict(type='xavier', distribution='uniform')``.
        ffn_init_cfg (dict | str | None, optional): The initialization config
            for linear layers in the feed forward network. Default:
            ``dict(type='kaiming')``.
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
                 order=('cross_att', 'ffn'),
                 att_init_cfg=dict(type='xavier', distribution='uniform'),
                 ffn_init_cfg=dict(type='kaiming')):
        super(CrossAttentionLayer, self).__init__()

        assert all(o in ('cross_att', 'ffn') for o in order)
        assert len(order) == len(set(order))

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
        self._order = order
        self._att_init_cfg = att_init_cfg
        self._ffn_init_cfg = ffn_init_cfg

        for name in order:
            if name == 'cross_att':
                self.b_to_a_att = MultiHeadAttention(
                    dims,
                    heads=heads,
                    att_dropout=att_dropout,
                    out_dropout=att_out_dropout,
                    bias=bias,
                    init_cfg=att_init_cfg)
                self.a_to_b_att = MultiHeadAttention(
                    dims,
                    heads=heads,
                    att_dropout=att_dropout,
                    out_dropout=att_out_dropout,
                    bias=bias,
                    init_cfg=att_init_cfg)
                self.b_to_a_att_norm = build_norm_layer(norm_cfg, dims=dims)
                self.a_to_b_att_norm = build_norm_layer(norm_cfg, dims=dims)
            else:
                self.a_ffn = FeedForwardNetwork(
                    dims,
                    ratio=ratio,
                    ffn_dropout=ffn_dropout,
                    out_dropout=ffn_out_dropout,
                    act_cfg=act_cfg,
                    init_cfg=ffn_init_cfg)
                self.b_ffn = FeedForwardNetwork(
                    dims,
                    ratio=ratio,
                    ffn_dropout=ffn_dropout,
                    out_dropout=ffn_out_dropout,
                    act_cfg=act_cfg,
                    init_cfg=ffn_init_cfg)
                self.a_ffn_norm = build_norm_layer(norm_cfg, dims=dims)
                self.b_ffn_norm = build_norm_layer(norm_cfg, dims=dims)

        if droppath > 0:
            self.droppath = DropPath(p=droppath)

    def forward(self, a, b, a_mask=None, b_mask=None):
        _a, _b = a, b

        if self._pre_norm:
            for name in self._order:
                if name == 'cross_att':
                    q = self.b_to_a_att_norm(a)
                    d = self.b_to_a_att(q, _b, _b, mask=b_mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    a = a + d

                    q = self.a_to_b_att_norm(b)
                    d = self.a_to_b_att(q, _a, _a, mask=a_mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    b = b + d

                    _a, _b = a, b
                else:
                    d = self.a_ffn_norm(a)
                    d = self.a_ffn(d)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    a = a + d

                    d = self.b_ffn_norm(b)
                    d = self.b_ffn(d)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    b = b + d
        else:
            for name in self._order:
                if name == 'cross_att':
                    d = self.b_to_a_att(a, _b, _b, mask=b_mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    a = self.b_to_a_att_norm(a + d)

                    d = self.a_to_b_att(b, _a, _a, mask=a_mask)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    b = self.a_to_b_att_norm(b + d)

                    _a, _b = a, b
                else:
                    d = self.a_ffn(a)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    a = self.a_ffn_norm(a + d)

                    d = self.b_ffn(b)
                    if self._droppath > 0:
                        d = self.droppath(d)
                    b = self.b_ffn_norm(b + d)

        return a, b
