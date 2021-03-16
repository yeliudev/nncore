# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GHMCLoss(nn.Module):
    """
    Loss introduced by the paper 'Gradient Harmonized Single-stage Detector':
        https://arxiv.org/abs/1811.05181

    Args:
        bins (int, optional): number of the unit regions for distribution
            calculation
        momentum (float, optional): the parameter for moving average
    """

    def __init__(self, bins=10, momentum=0):
        super(GHMCLoss, self).__init__()

        self.bins = bins
        self.momentum = momentum

        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)

    def forward(self, pred, target):
        weights = torch.zeros_like(pred)
        g = torch.abs(pred.sigmoid().detach() - target)

        tot = target.size(1)
        n = 0
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] + (
                        1 - self.momentum) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot

        return loss
