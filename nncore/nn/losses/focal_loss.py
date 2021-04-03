# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import nncore


def sigmoid_focal_loss(pred, target, alpha=-1, gamma=2, reduction='mean'):
    """
    Focal Loss introduced in [1].

    Args:
        pred (:obj:`torch.Tensor`): The predictions for each example.
        target (:obj:`torch.Tensor`): The binary classification label for
            each element (0 for negative classes and 1 for positive classes).
        alpha (int or float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive vs negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (int or float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy vs hard examples. Default: ``2``.
        reduction (str, optional): Reduction method. Currently supported
            methods include ``'mean'``, ``'sum'`` and ``'none'``. Default:
            ``'mean'``.

    Returns:
        :obj:`torch.Tensor`: The loss tensor with reduction option applied.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """
    p = torch.sigmoid(pred)
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


def sigmoid_focal_loss_star(pred, target, alpha=-1, gamma=1, reduction='mean'):
    """
    Focal Loss* introduced in [1].

    Args:
        pred (:obj:`torch.Tensor`): The predictions for each example.
        target (:obj:`torch.Tensor`): The binary classification label for
            each element (0 for negative classes and 1 for positive classes).
        alpha (int or float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive vs negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (int or float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy vs hard examples. Default: ``2``.
        reduction (str, optional): Reduction method. Currently supported
            methods include ``'mean'``, ``'sum'`` and ``'none'``. Default:
            ``'mean'``.

    Returns:
        :obj:`torch.Tensor`: The loss tensor with reduction option applied.

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


@nncore.bind_getter('alpha', 'gamma', 'reduction', 'loss_weight')
class FocalLoss(nn.Module):
    """
    Focal Loss introduced in [1].

    Args:
        alpha (int or float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive vs negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (int or float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy vs hard examples. Default: ``2``.
        reduction (str, optional): Reduction method. Currently supported
            methods include ``'mean'``, ``'sum'`` and ``'none'``. Default:
            ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1``.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self, alpha=-1, gamma=2, reduction='mean', loss_weight=1):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._loss_weight = loss_weight

    def extra_repr(self):
        return 'alpha={}, gamma={}, reduction={}, loss_weight={}'.format(
            self._alpha, self._gamma, self._reduction, self._loss_weight)

    def forward(self, pred, target):
        return sigmoid_focal_loss(
            pred,
            target,
            alpha=self._alpha,
            gamma=self._gamma,
            reduction=self._reduction) * self._loss_weight


@nncore.bind_getter('alpha', 'gamma', 'reduction', 'loss_weight')
class FocalLossStar(FocalLoss):
    """
    Focal Loss* introduced in [1].

    Args:
        alpha (int or float, optional): Weighting factor in range ``(0, 1)`` to
            balance positive vs negative examples. ``-1`` means no weighting.
            Default: ``-1``.
        gamma (int or float, optional): Exponent of the modulating factor
            ``(1 - p_t)`` to balance easy vs hard examples. Default: ``2``.
        reduction (str, optional): Reduction method. Currently supported
            methods include ``'mean'``, ``'sum'`` and ``'none'``. Default:
            ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1``.

    References:
        1. Lin et al. (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self, alpha=-1, gamma=1, reduction='mean', loss_weight=1):
        super(FocalLossStar, self).__init__(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            loss_weight=loss_weight)

    def forward(self, pred, target):
        return sigmoid_focal_loss_star(
            pred,
            target,
            alpha=self._alpha,
            gamma=self._gamma,
            reduction=self._reduction) * self._loss_weight
