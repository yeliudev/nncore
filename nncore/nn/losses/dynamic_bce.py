# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F

import nncore
from ..builder import LOSSES


@LOSSES.register()
@nncore.bind_getter('reduction', 'pos_weight', 'loss_weight')
class DynamicBCELoss(nn.Module):
    """
    Dynamic Binary Cross Entropy Loss that supports dynamic loss weights
    during training.

    Args:
        reduction (str, optional): Reduction method. Currently supported values
            include ``'mean'``, ``'sum'``, and ``'none'``. Default: ``'mean'``.
        pos_weight (float | None, optional): Weight of the positive examples.
            Default: ``None``.
        loss_weight (float, optional): Weight of the loss. Default: ``1.0``.
    """

    def __init__(self, reduction='mean', pos_weight=None, loss_weight=1.0):
        super(DynamicBCELoss, self).__init__()
        assert reduction in ('mean', 'sum', 'none')

        self._reduction = reduction
        self._pos_weight = pos_weight
        self._loss_weight = loss_weight

    def extra_repr(self):
        return "reduction='{}', pos_weight={}, loss_weight={}".format(
            self._reduction, self._pos_weight, self._loss_weight)

    def forward(self, pred, target, weight=None):
        if self._pos_weight is not None:
            pos_weight = pred.new_tensor([self._pos_weight] * pred.size(1))
        else:
            pos_weight = None

        loss = F.binary_cross_entropy_with_logits(
            pred,
            target,
            weight=weight,
            reduction=self._reduction,
            pos_weight=pos_weight)

        return loss * self._loss_weight
