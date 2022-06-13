# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch
from torch.distributions import Gumbel


def hard_softmax(x, dim=-1):
    """
    Compute hard softmax across a specified dimension.

    Args:
        x (:obj:`torch.Tensor`): The input tensor.
        dim (int, optional): The dimension to be computed. Default: ``-1``.

    Returns:
        :obj:`torch.Tensor`: The computed binary tensor.
    """
    soft = x.softmax(dim)
    inds = soft.argmax(dim, keepdim=True)

    hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format)
    hard.scatter_(dim, inds, 1.0)

    x = hard + soft - soft.detach()
    return x


def gumbel_softmax(x, tau=1.0, hard_assign=True, dim=-1):
    """
    Compute gumbel softmax across a specified dimension.

    Args:
        x (:obj:`torch.Tensor`): The input tensor.
        tau (float, optional): The parameter of gumbel softmax. Default:
            ``1.0``.
        hard_assign (bool, optional): Whether to apply hard assignment
            strategy. Default: ``True``.
        dim (int, optional): The dimension to be computed. Default: ``-1``.

    Returns:
        :obj:`torch.Tensor`: The computed binary tensor.
    """
    dist = Gumbel(x.new_tensor(0), x.new_tensor(1))
    gumbel = dist.sample(x.shape)

    gumbel = (x + gumbel) / tau
    soft = gumbel.softmax(dim)

    if hard_assign:
        inds = soft.argmax(dim, keepdim=True)
        hard = torch.zeros_like(
            x, memory_format=torch.legacy_contiguous_format)
        hard.scatter_(dim, inds, 1.0)
        x = hard + soft - soft.detach()
    else:
        x = soft

    return x
