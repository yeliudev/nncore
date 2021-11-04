# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F

import nncore
from ..builder import MODELS
from .bundle import Parameter


@MODELS.register()
@nncore.bind_getter('q_dims', 'k_dims', 'v_dims', 'heads', 'p', 'bias')
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module introduced in [1].

    Args:
        dims (list[int] | int): The feature dimensions of the model. If it is
            a list, it is expected to have 3 ``int`` values indicating the
            dimensions of query, key, and value matrices.
        heads (int, optional): The number of attention heads. Default: ``1``.
        p (float, optional): The probability in dropout layer. Default: ``0``.
        bias (bool, optional): Whether to add the bias term. Default: ``True``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, dims, heads=1, p=0, bias=True):
        super(MultiHeadAttention, self).__init__()

        if isinstance(dims, int):
            dims = [dims] * 3

        self._q_dims, self._k_dims, self._v_dims = dims
        self._same_dims = dims[0] == dims[1] == dims[2]
        self._heads = heads
        self._p = p
        self._bias = bias

        if self._same_dims:
            self.weight_i = Parameter(self._q_dims * 3, self._q_dims)
            self.register_parameter('weight_q', None)
            self.register_parameter('weight_k', None)
            self.register_parameter('weight_v', None)
        else:
            self.weight_q = Parameter(self._q_dims, self._q_dims)
            self.weight_k = Parameter(self._q_dims, self._k_dims)
            self.weight_v = Parameter(self._q_dims, self._v_dims)
            self.register_parameter('weight_i', None)

        self.weight_o = Parameter(self._q_dims, self._q_dims)

        if bias:
            self.bias_i = Parameter(self._q_dims * 3)
            self.bias_o = Parameter(self._q_dims)
        else:
            self.register_parameter('bias_i', None)
            self.register_parameter('bias_o', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self._same_dims:
            nn.init.xavier_uniform_(self.weight_i)
        else:
            nn.init.xavier_uniform_(self.weight_q)
            nn.init.xavier_uniform_(self.weight_k)
            nn.init.xavier_uniform_(self.weight_v)

        nn.init.xavier_uniform_(self.weight_o)

        if self.bias_i is not None:
            nn.init.constant_(self.bias_i, 0)

        if self.bias_o is not None:
            nn.init.constant_(self.bias_o, 0)

    def forward(self, query, key=None, value=None, mask=None, **kwargs):
        out, _ = F.multi_head_attention_forward(
            query,
            key or query,
            value or query,
            self._q_dims,
            self._heads,
            self.weight_i,
            self.bias_i,
            None,
            None,
            False,
            self._p,
            self.weight_o,
            self.bias_o,
            training=self.training,
            need_weights=False,
            attn_mask=mask,
            use_separate_proj_weight=not self._same_dims,
            q_proj_weight=self.weight_q,
            k_proj_weight=self.weight_k,
            v_proj_weight=self.weight_v,
            **kwargs)
        return out
