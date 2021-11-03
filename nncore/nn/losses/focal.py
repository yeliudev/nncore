# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F

import nncore
from ..builder import LOSSES


def focal_loss(pred, target, alpha=-1, gamma=2.0, reduction='mean'):
    """
    Focal Loss introduced in [1].

    Args:
        pred (:obj:`torch.Tensor`): The predictions.
        target (:obj:`torch.Tensor`): The binary classification label for
            each element (0 for negative classes and 1 for positive classes).
        alpha (int | float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``2.0``.
        reduction (str, optional): Reduction method. Currently supported
            values include ``'mean'``, ``'sum'``, and ``'none'``. Default:
            ``'mean'``.

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

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


def focal_loss_star(pred, target, alpha=-1, gamma=1.0, reduction='mean'):
    """
    Focal Loss* introduced in [1].

    Args:
        pred (:obj:`torch.Tensor`): The predictions.
        target (:obj:`torch.Tensor`): The binary classification label for
            each element (0 for negative classes and 1 for positive classes).
        alpha (int | float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``1.0``.
        reduction (str, optional): Reduction method. Currently supported
            values include ``'mean'``, ``'sum'``, and ``'none'``. Default:
            ``'mean'``.

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

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


@LOSSES.register()
@nncore.bind_getter('alpha', 'gamma', 'reduction', 'loss_weight')
class FocalLoss(nn.Module):
    """
    Focal Loss introduced in [1].

    Args:
        alpha (int | float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``2.0``.
        reduction (str, optional): Reduction method. Currently supported
            values include ``'mean'``, ``'sum'``, and ``'none'``. Default:
            ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self, alpha=-1, gamma=2.0, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert reduction in ('mean', 'sum', 'none')

        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._loss_weight = loss_weight

    def extra_repr(self):
        return 'alpha={}, gamma={}, reduction={}, loss_weight={}'.format(
            self._alpha, self._gamma, self._reduction, self._loss_weight)

    def forward(self, pred, target):
        return focal_loss(
            pred,
            target,
            alpha=self._alpha,
            gamma=self._gamma,
            reduction=self._reduction) * self._loss_weight


@LOSSES.register()
@nncore.bind_getter('alpha', 'gamma', 'reduction', 'loss_weight')
class FocalLossStar(FocalLoss):
    """
    Focal Loss* introduced in [1].

    Args:
        alpha (int | float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive and negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy and hard examples. Default: ``1.0``.
        reduction (str, optional): Reduction method. Currently supported
            values include ``'mean'``, ``'sum'``, and ``'none'``. Default:
            ``'mean'``.
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

    def forward(self, pred, target):
        return focal_loss_star(
            pred,
            target,
            alpha=self._alpha,
            gamma=self._gamma,
            reduction=self._reduction) * self._loss_weight
