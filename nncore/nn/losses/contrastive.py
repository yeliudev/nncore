# Copyright (c) Ye Liu. Licensed under the MIT License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import nncore
from nncore.ops import cosine_similarity
from ..builder import LOSSES
from ..bundle import Parameter
from .utils import weighted_loss


@weighted_loss
def infonce_loss(a, b, temperature=0.07, scale=None, max_scale=100):
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

    References:
        1. Oord et al. (https://arxiv.org/abs/1807.03748)
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    if scale is None:
        scale = a.new_tensor([math.log(1 / temperature)])

    scale = scale.exp().clamp(max=max_scale)
    a_sim = torch.matmul(a, b.transpose(-1, -2)) * scale
    b_sim = a_sim.transpose(-1, -2)

    target = torch.arange(a.size(-2), device=a.device).expand(a.size()[:-1])
    a_loss = F.cross_entropy(a_sim, target)
    b_loss = F.cross_entropy(b_sim, target)

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
            Default: ``True``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.

    References:
        1. Oord et al. (https://arxiv.org/abs/1807.03748)
    """

    def __init__(self,
                 temperature=0.07,
                 max_scale=100,
                 learnable=True,
                 loss_weight=1.0):
        super(InfoNCELoss, self).__init__()

        if learnable:
            self.scale = Parameter(math.log(1 / temperature))
        else:
            self.scale = None

        self._temperature = temperature
        self._max_scale = max_scale
        self._learnable = learnable
        self._loss_weight = loss_weight

    def extra_repr(self):
        return ('temperature={}, max_scale={}, learnable={}, loss_weight={}'.
                format(self._temperature, self._max_scale, self._learnable,
                       self._loss_weight))

    def forward(self, a, b, weight=None, avg_factor=None):
        return infonce_loss(
            a,
            b,
            temperature=self._temperature,
            scale=self.scale,
            max_scale=self._max_scale,
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
        return 'margin={}, reduction={}, loss_weight={}'.format(
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
