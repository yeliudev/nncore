# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import nncore


@nncore.bind_getter('bins', 'momentum', 'loss_weight')
class GHMCLoss(nn.Module):
    """
    Gradient Harmonized Classification Loss introduced in [1].

    Args:
        bins (int, optional): Number of the unit regions for distribution
            calculation. Default: ``10``.
        momentum (float, optional): The parameter for moving average. Default:
            ``0``.
        loss_weight (float, optional): Weight of the loss. Default: ``1``.

    References:
        1. Li et al. (https://arxiv.org/abs/1811.05181)
    """

    def __init__(self, bins=10, momentum=0, loss_weight=1):
        super(GHMCLoss, self).__init__()
        self._bins = bins
        self._momentum = momentum
        self._loss_weight = loss_weight

        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)

    def extra_repr(self):
        return 'bins={}, momentum={}, loss_weight={}'.format(
            self._bins, self._momentum, self._loss_weight)

    def forward(self, pred, target):
        weights = torch.zeros_like(pred)
        g = torch.abs(pred.sigmoid().detach() - target)

        tot = target.size(1)
        n = 0
        for i in range(self._bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bins = inds.sum().item()
            if num_in_bins > 0:
                if self._momentum > 0:
                    self.acc_sum[i] = self._momentum * self.acc_sum[i] + (
                        1 - self._momentum) * num_in_bins
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bins
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot

        loss *= self._loss_weight
        return loss
