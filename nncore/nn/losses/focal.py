# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F

import nncore
from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def focal_loss(pred, target, alpha=-1, gamma=2.0):
    """
    Focal Loss introduced in [1].

    Args:
        pred (:obj:`torch.Tensor`): The predictions.
        target (:obj:`torch.Tensor`): The binary classification label for
            each element (0 for negative classes and 1 for positive classes).
        alpha (float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``2.0``.

    Returns:
        :obj:`torch.Tensor`: The loss tensor.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """
    p = pred.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    return loss


@weighted_loss
def focal_loss_star(pred, target, alpha=-1, gamma=1.0):
    """
    Focal Loss* introduced in [1].

    Args:
        pred (:obj:`torch.Tensor`): The predictions.
        target (:obj:`torch.Tensor`): The binary classification label for
            each element (0 for negative classes and 1 for positive classes).
        alpha (float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``1.0``.

    Returns:
        :obj:`torch.Tensor`: The loss tensor.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """
    shifted_inputs = gamma * (pred * (2 * target - 1))
    loss = -F.logsigmoid(shifted_inputs) / gamma

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss *= alpha_t

    return loss


@weighted_loss
def gaussian_focal_loss(pred, target, alpha=2.0, gamma=4.0):
    """
    Focal Loss introduced in [1] for targets in gaussian distribution.

    Args:
        pred (:obj:`torch.Tensor`): The predictions.
        target (:obj:`torch.Tensor`): The learning targets in gaussian
            distribution.
        alpha (float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``2.0``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``4.0``.

    Returns:
        :obj:`torch.Tensor`: The loss tensor.

    References:
        1. Law et al. (https://arxiv.org/abs/1808.01244)
    """
    eps = 1e-12

    pos_weights = target.eq(1)
    neg_weights = (1 - target).pow(gamma)

    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights

    loss = pos_loss + neg_loss
    return loss


@LOSSES.register()
@nncore.bind_getter('alpha', 'gamma', 'reduction', 'loss_weight')
class FocalLoss(nn.Module):
    """
    Focal Loss introduced in [1].

    Args:
        alpha (float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``2.0``.
        reduction (str, optional): Reduction method. Currently supported values
            include ``'mean'``, ``'sum'``, and ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self, alpha=-1, gamma=2.0, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._loss_weight = loss_weight

    def extra_repr(self):
        return "alpha={}, gamma={}, reduction='{}', loss_weight={}".format(
            self._alpha, self._gamma, self._reduction, self._loss_weight)

    def forward(self, pred, target, weight=None, avg_factor=None):
        return focal_loss(
            pred,
            target,
            alpha=self._alpha,
            gamma=self._gamma,
            weight=weight,
            reduction=self._reduction,
            avg_factor=avg_factor) * self._loss_weight


@LOSSES.register()
@nncore.bind_getter('alpha', 'gamma', 'reduction', 'loss_weight')
class FocalLossStar(FocalLoss):
    """
    Focal Loss* introduced in [1].

    Args:
        alpha (float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``1.0``.
        reduction (str, optional): Reduction method. Currently supported values
            include ``'mean'``, ``'sum'``, and ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self, alpha=-1, gamma=1.0, reduction='mean', loss_weight=1.0):
        super(FocalLossStar, self).__init__(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            loss_weight=loss_weight)

    def forward(self, pred, target, weight=None, avg_factor=None):
        return focal_loss_star(
            pred,
            target,
            alpha=self._alpha,
            gamma=self._gamma,
            weight=weight,
            reduction=self._reduction,
            avg_factor=avg_factor) * self._loss_weight


@LOSSES.register()
@nncore.bind_getter('alpha', 'gamma', 'reduction', 'loss_weight')
class GaussianFocalLoss(FocalLoss):
    """
    Focal Loss introduced in [1] for targets in gaussian distribution.

    Args:
        alpha (float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``2.0``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``4.0``.
        reduction (str, optional): Reduction method. Currently supported values
            include ``'mean'``, ``'sum'``, and ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            loss_weight=loss_weight)

    def forward(self, pred, target, weight=None, avg_factor=None):
        return gaussian_focal_loss(
            pred,
            target,
            alpha=self._alpha,
            gamma=self._gamma,
            weight=weight,
            reduction=self._reduction,
            avg_factor=avg_factor) * self._loss_weight
