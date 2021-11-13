# Copyright (c) Ye Liu. All rights reserved.

import numpy as np
import torch
import torch.nn as nn

import nncore
from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def l1_loss(pred, target):
    """
    L1 Loss.

    Args:
        pred (:obj:`torch.Tensor`): The predictions.
        target (:obj:`torch.Tensor`): The learning targets.

    Returns:
        :obj:`torch.Tensor`: The loss tensor.
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = (pred - target).abs()

    return loss


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """
    Smooth L1 Loss introduced in [1].

    Args:
        pred (:obj:`torch.Tensor`): The predictions.
        target (:obj:`torch.Tensor`): The learning targets.
        beta (float, optional): The threshold in the piecewise function.
            Default: ``1.0``.

    Returns:
        :obj:`torch.Tensor`: The loss tensor.

    References:
        1. Girshick et al. (https://arxiv.org/abs/1504.08083)
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert beta > 0
    assert pred.size() == target.size()

    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    return loss


@weighted_loss
def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5):
    """
    Balanced L1 Loss introduced in [1].

    Args:
        pred (:obj:`torch.Tensor`): The predictions.
        target (:obj:`torch.Tensor`): The learning targets.
        beta (float, optional): The threshold in the piecewise function.
            Default: ``1.0``.
        alpha (float, optional): The dominator of the loss. Default: ``0.5``.
        gamma (float, optional): The promotion controller of the loss. Default:
            ``1.5``.

    Returns:
        :obj:`torch.Tensor`: The loss tensor.

    References:
        1. Pang et al. (https://arxiv.org/abs/1904.02701)
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert beta > 0
    assert pred.size() == target.size()

    b = np.e**(gamma / alpha) - 1
    diff = (pred - target).abs()
    loss = torch.where(
        diff < beta, alpha / b * (b * diff + 1) * (b * diff / beta + 1).log() -
        alpha * diff, gamma * diff + gamma / b - alpha * beta)

    return loss


@LOSSES.register()
@nncore.bind_getter('reduction', 'loss_weight')
class L1Loss(nn.Module):
    """
    L1 Loss.

    Args:
        reduction (str, optional): Reduction method. Currently supported values
            include ``'mean'``, ``'sum'``, and ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self._reduction = reduction
        self._loss_weight = loss_weight

    def extra_repr(self):
        return "reduction='{}', loss_weight={}".format(self._reduction,
                                                       self._loss_weight)

    def forward(self, pred, target, weight=None, avg_factor=None):
        return l1_loss(
            pred,
            target,
            weight=weight,
            reduction=self._reduction,
            avg_factor=avg_factor) * self._loss_weight


@LOSSES.register()
@nncore.bind_getter('beta', 'reduction', 'loss_weight')
class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss introduced in [1].

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Default: ``1.0``.
        reduction (str, optional): Reduction method. Currently supported values
            include ``'mean'``, ``'sum'``, and ``'none'``. Default: ``'mean'``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.

    References:
        1. Girshick et al. (https://arxiv.org/abs/1504.08083)
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self._beta = beta
        self._reduction = reduction
        self._loss_weight = loss_weight

    def extra_repr(self):
        return "beta={}, reduction='{}', loss_weight={}".format(
            self._beta, self._reduction, self._loss_weight)

    def forward(self, pred, target, weight=None, avg_factor=None):
        return smooth_l1_loss(
            pred,
            target,
            beta=self._beta,
            weight=weight,
            reduction=self._reduction,
            avg_factor=avg_factor) * self._loss_weight


@LOSSES.register()
@nncore.bind_getter('beta', 'alpha', 'gamma', 'reduction', 'loss_weight')
class BalancedL1Loss(nn.Module):
    """
    Balanced L1 Loss introduced in [1].

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Default: ``1.0``.
        alpha (float, optional): The dominator of the loss. Default: ``0.5``.
        gamma (float, optional): The promotion controller of the loss. Default:
            ``1.5``.
        reduction (str, optional): Reduction method. Currently supported values
            include ``'mean'``, ``'sum'``, and ``'none'``. Default: ``'mean'``.

    References:
        1. Pang et al. (https://arxiv.org/abs/1904.02701)
    """

    def __init__(self,
                 beta=1.0,
                 alpha=0.5,
                 gamma=1.5,
                 reduction='mean',
                 loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self._beta = beta
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._loss_weight = loss_weight

    def extra_repr(self):
        return ("beta={}, alpha={}, gamma={}, reduction='{}', "
                "loss_weight={}".format(self._beta, self._alpha, self._gamma,
                                        self._reduction, self._loss_weight))

    def forward(self, pred, target, weight=None, avg_factor=None):
        return balanced_l1_loss(
            pred,
            target,
            beta=self._beta,
            alpha=self._alpha,
            gamma=self._gamma,
            weight=weight,
            reduction=self._reduction,
            avg_factor=avg_factor) * self._loss_weight
