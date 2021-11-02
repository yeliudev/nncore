# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import nncore
from ..builder import ACTIVATIONS


class _MishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * F.softplus(i).tanh()
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1. / h.cosh().pow_(2)
        grad_hx = i.sigmoid()
        grad_gx = grad_gh * grad_hx
        grad_f = grad_gx * i + F.softplus(i).tanh()
        return grad_output * grad_f


class _SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * i.sigmoid()
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = i.sigmoid()
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


@ACTIVATIONS.register()
class EffMish(nn.Module):
    """
    An efficient implementation of Mish activation layer introduced in [1].

    References:
        1. Misra et al. (https://arxiv.org/abs/1908.08681)
    """

    def forward(self, x):
        return _MishImplementation.apply(x)


@ACTIVATIONS.register()
class EffSwish(nn.Module):
    """
    An efficient implementation of Swish activation layer introduced in [1].

    References:
        1. Ramachandran et al. (https://arxiv.org/abs/1710.05941)
    """

    def forward(self, x):
        return _SwishImplementation.apply(x)


@ACTIVATIONS.register()
class Mish(nn.Module):
    """
    Mish activation layer introduced in [1].

    References:
        1. Misra et al. (https://arxiv.org/abs/1908.08681)
    """

    def forward(self, x):
        return x * F.softplus(x).tanh()


@ACTIVATIONS.register()
class Swish(nn.Module):
    """
    Swish activation layer introduced in [1].

    References:
        1. Ramachandran et al. (https://arxiv.org/abs/1710.05941)
    """

    def forward(self, x):
        return x * x.sigmoid()


@ACTIVATIONS.register()
@nncore.bind_getter('min', 'max')
class Clamp(nn.Module):
    """
    Clamp activation layer.

    Args:
        min (float, optional): The lower-bound of the range. Default: ``-1``.
        max (float, optional): The upper-bound of the range. Default: ``1``.
    """

    def __init__(self, min=-1, max=1):
        super(Clamp, self).__init__()
        self._min = min
        self._max = max

    def forward(self, x):
        return x.clamp(self._min, self._max)
