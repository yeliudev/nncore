# Copyright (c) Ye Liu. Licensed under the MIT License.

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import nncore
from nncore.ops import cosine_similarity
from ..builder import LOSSES
from ..bundle import Parameter
from .utils import weighted_loss


class _AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, group=None):
        ctx.size = tensor.size(-2)
        ctx.rank = dist.get_rank(group=group)
        ctx.group = group

        world_size = dist.get_world_size(group=group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=group)

        gathered = torch.cat(gathered, dim=-2)
        return gathered

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad[..., ctx.size * ctx.rank:ctx.size * (ctx.rank + 1), :]


@weighted_loss
def infonce_loss(a,
                 b,
                 temperature=0.07,
                 scale=None,
                 max_scale=100,
                 dist=False,
                 group=None):
    """
    InfoNCE Loss introduced in [1].

    Args:
        a (:obj:`torch.Tensor`): The first group of samples.
        b (:obj:`torch.Tensor`): The second group of samples.
        temperature (float, optional): The temperature for softmax. Default:
            ``0.07``.
        scale (:obj:`torch.Tensor` | None, optional): The logit scale to use.
            If not specified, the scale will be calculated from temperature.
            Default: ``None``.
        max_scale (float, optional): The maximum logit scale value. Default:
            ``100``.
        dist (bool, optional): Whether the loss is computed across processes.
            Default: ``False``.
        group (:obj:`dist.ProcessGroup` | None, optional): The process group
            to use. If not specified, the default process group will be used.
            Default: ``None``.

    References:
        1. Oord et al. (https://arxiv.org/abs/1807.03748)
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    if scale is None:
        scale = a.new_tensor([math.log(1 / temperature)])

    scale = scale.exp().clamp(max=max_scale)

    n = a.size(-2)
    t = torch.arange(n, device=a.device)

    if dist and dist.is_initialized():
        rank = dist.get_rank(group=group)

        s, e = n * rank, n * (rank + 1)
        t += s

        a = _AllGather.apply(a)
        b = _AllGather.apply(b)

        a_sim = torch.matmul(a[..., s:e, :], b.transpose(-1, -2)) * scale
        b_sim = torch.matmul(b[..., s:e, :], a.transpose(-1, -2)) * scale
    else:
        a_sim = torch.matmul(a, b.transpose(-1, -2)) * scale
        b_sim = a_sim.transpose(-1, -2).contiguous()

    a_sim = a_sim.view(-1, a_sim.size(-1))
    b_sim = b_sim.view(-1, b_sim.size(-1))

    a_loss = F.cross_entropy(a_sim, t.repeat(int(a_sim.size(0) / n)))
    b_loss = F.cross_entropy(b_sim, t.repeat(int(b_sim.size(0) / n)))

    loss = (a_loss + b_loss) / 2
    return loss


@weighted_loss
def triplet_loss(pos, neg, anchor, margin=0.5):
    """
    Triplet Loss.

    Args:
        pos (:obj:`torch.Tensor`): Positive samples.
        neg (:obj:`torch.Tensor`): Negative samples.
        anchor (:obj:`torch.Tensor`): Anchors for distance calculation.
        margin (float, optional): The margin between positive and negative
            samples. Default: ``0.5``.

    Returns:
        :obj:`torch.Tensor`: The loss tensor.
    """
    pos_sim = cosine_similarity(pos, anchor)
    neg_sim = cosine_similarity(neg, anchor)

    loss = (margin - pos_sim + neg_sim).relu()
    return loss


@LOSSES.register()
@nncore.bind_getter('temperature', 'max_scale', 'learnable', 'loss_weight')
class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss introduced in [1].

    Args:
        temperature (float, optional): The initial temperature for softmax.
            Default: ``0.07``.
        max_scale (float, optional): The maximum value of learnable scale.
            Default: ``100``.
        learnable (bool, optional): Whether the logit scale is learnable.
            Default: ``False``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.

    References:
        1. Oord et al. (https://arxiv.org/abs/1807.03748)
    """

    def __init__(self,
                 temperature=0.07,
                 max_scale=100,
                 learnable=False,
                 dist=False,
                 loss_weight=1.0):
        super(InfoNCELoss, self).__init__()

        if learnable:
            self.scale = Parameter(math.log(1 / temperature))
        else:
            self.scale = None

        self._temperature = temperature
        self._max_scale = max_scale
        self._learnable = learnable
        self._dist = dist
        self._loss_weight = loss_weight

    def extra_repr(self):
        return ('temperature={}, max_scale={}, learnable={}, dist={}, '
                'loss_weight={}'.format(self._temperature, self._max_scale,
                                        self._learnable, self._dist,
                                        self._loss_weight))

    def forward(self, a, b, weight=None, avg_factor=None):
        return infonce_loss(
            a,
            b,
            temperature=self._temperature,
            scale=self.scale,
            max_scale=self._max_scale,
            dist=self._dist,
            weight=weight,
            avg_factor=avg_factor) * self._loss_weight


@LOSSES.register()
@nncore.bind_getter('margin', 'reduction', 'loss_weight')
class TripletLoss(nn.Module):
    """
    Triplet Loss.

    Args:
        margin (float, optional): The margin between positive and negative
            samples. Default: ``0.5``.
        reduction (str, optional): Reduction method. Currently supported values
            include ``'mean'``, ``'sum'``, and ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.
    """

    def __init__(self, margin=0.5, reduction='mean', loss_weight=1.0):
        super(TripletLoss, self).__init__()

        self._margin = margin
        self._reduction = reduction
        self._loss_weight = loss_weight

    def extra_repr(self):
        return "margin={}, reduction='{}', loss_weight={}".format(
            self._margin, self._reduction, self._loss_weight)

    def forward(self, pos, neg, anchor, weight=None, avg_factor=None):
        return triplet_loss(
            pos,
            neg,
            anchor,
            margin=self._margin,
            weight=weight,
            reduction=self._reduction,
            avg_factor=avg_factor) * self._loss_weight
