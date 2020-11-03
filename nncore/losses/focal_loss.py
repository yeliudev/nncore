# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(pred, target, alpha=-1, gamma=2, reduction='mean'):
    """
    Loss used in RetinaNet for dense object detection:
        https://arxiv.org/abs/1708.02002.

    Args:
        inputs (:obj:`torch.Tensor`): the predictions for each example
        targets (:obj:`torch.Tensor`): the binary classification label for
            each element (0 for negative classes and 1 for positive classes)
        alpha (int or float, optional): weighting factor in range (0, 1) to
            balance positive vs negative examples. -1 means no weighting.
        gamma (int or float, optional): exponent of the modulating factor
            (1 - p_t) to balance easy vs hard examples
        reduction (str, optional): reduction methods. Currently supported
            methods include `none`, `mean`, and `sum`.

    Returns:
        loss (:obj:`torch.Tensor`): the loss tensor with reduction option
            applied
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
    FL* described in RetinaNet paper appendix:
        https://arxiv.org/abs/1708.02002.

    Args:
        inputs (:obj:`torch.Tensor`): the predictions for each example
        targets (:obj:`torch.Tensor`): the binary classification label for
            each element (0 for negative classes and 1 for positive classes)
        alpha (int or float, optional): weighting factor in range (0, 1) to
            balance positive vs negative examples. -1 means no weighting.
        gamma (int or float, optional): gamma parameter described in FL*.
            -1 means no weighting.
        reduction (str, optional): reduction methods. Currently supported
            methods include `none`, `mean`, and `sum`.

    Returns:
        loss (:obj:`torch.Tensor`): the loss tensor with reduction option
            applied
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


class FocalLoss(nn.Module):
    """
    Loss used in RetinaNet for dense object detection:
        https://arxiv.org/abs/1708.02002.

    Args:
        alpha (int or float, optional): weighting factor in range (0, 1) to
            balance positive vs negative examples. -1 means no weighting.
        gamma (int or float, optional): exponent of the modulating factor
            (1 - p_t) to balance easy vs hard examples
        reduction (str, optional): reduction methods. Currently supported
            methods include `none`, `mean`, and `sum`.
    """

    def __init__(self, alpha=-1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        return sigmoid_focal_loss(pred, target, self.alpha, self.gamma,
                                  self.reduction)


class FocalLossStar(nn.Module):
    """
    FL* described in RetinaNet paper appendix:
        https://arxiv.org/abs/1708.02002.

    Args:
        alpha (int or float, optional): weighting factor in range (0, 1) to
            balance positive vs negative examples. -1 means no weighting.
        gamma (int or float, optional): gamma parameter described in FL*.
            -1 means no weighting.
        reduction (str, optional): reduction methods. Currently supported
            methods include `none`, `mean`, and `sum`.
    """

    def __init__(self, alpha=-1, gamma=1, reduction='mean'):
        super(FocalLossStar, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        return sigmoid_focal_loss_star(pred, target, self.alpha, self.gamma,
                                       self.reduction)
