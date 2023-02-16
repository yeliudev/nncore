# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch
import torch.nn.functional as F
from torch.distributions import Gumbel


def cosine_similarity(x, y):
    """
    Compute the cosine similarities among two batches of tensors.

    Args:
        x (:obj:`torch.Tensor[*, N, C]`): The first batch of tensors.
        y (:obj:`torch.Tensor[*, M, C]`): The second batch of tensors.

    Returns:
        :obj:`torch.Tensor[*, N, M]`: The computed pairwise cosine \
            similarities.
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return torch.matmul(x, y.transpose(-1, -2))


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

    ret = hard + soft - soft.detach()
    return ret


def gumbel_softmax(x, tau=1.0, hard_assign=True, dim=-1):
    """
    Compute gumbel softmax across a specified dimension.

    Args:
        x (:obj:`torch.Tensor`): The input tensor.
        tau (float, optional): The temperature of gumbel softmax. Default:
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
    ret = gumbel.softmax(dim)

    if hard_assign:
        inds = ret.argmax(dim, keepdim=True)
        hard = torch.zeros_like(
            x, memory_format=torch.legacy_contiguous_format)
        hard.scatter_(dim, inds, 1.0)
        ret = hard + ret - ret.detach()

    return ret
