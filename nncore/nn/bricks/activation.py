# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn

import nncore

ACTIVATIONS = nncore.Registry('activation')


class _SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


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
class Swish(nn.Module):
    """
    Swish activation layer introduced in [1].

    References:
        1. Ramachandran et al. (https://arxiv.org/abs/1710.05941)
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


@ACTIVATIONS.register()
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
        return torch.clamp(x, min=self._min, max=self._max)


def build_act_layer(cfg, **kwargs):
    """
    Build an activation layer from a dict. This method searches for layers in
    :obj:`ACTIVATIONS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict or str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return nncore.build_object(cfg, [ACTIVATIONS, nn], **kwargs)
