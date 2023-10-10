# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch
import torch.nn as nn

import nncore
from ..builder import NORMS

NORMS.register(nn.BatchNorm1d, name='BN1d', group=['norm', '1d'])
NORMS.register(nn.BatchNorm2d, name=['BN', 'BN2d'], group=['norm', '2d'])
NORMS.register(nn.BatchNorm3d, name='BN3d', group=['norm', '3d'])
NORMS.register(nn.SyncBatchNorm, name='SyncBN', group=['norm', '2d'])
NORMS.register(nn.GroupNorm, name='GN', group=['norm', '2d'])
NORMS.register(nn.LayerNorm, name='LN', group=['norm', '1d', '2d', '3d'])
NORMS.register(nn.InstanceNorm1d, name='IN1d', group=['norm', '1d'])
NORMS.register(nn.InstanceNorm2d, name=['IN', 'IN2d'], group=['norm', '2d'])
NORMS.register(nn.InstanceNorm3d, name='IN3d', group=['norm', '3d'])
NORMS.register(nn.Dropout, name=['Drop1d', 'drop'], group=['drop', '1d'])
NORMS.register(nn.Dropout2d, name='Drop2d', group=['drop', '2d'])
NORMS.register(nn.Dropout3d, name='Drop3d', group=['drop', '3d'])
NORMS.register(nn.AlphaDropout, name='ADrop', group=['drop', '1d'])
NORMS.register(nn.FeatureAlphaDropout, name='FADrop', group=['drop', '1d'])


def drop_path(x, p=0.1, training=False):
    """
    DropPath operation introducted in [1].

    Args:
        p (float, optional): Probability of the path to be dropped. Default:
            ``0.1``.
        training (bool, optional): Whether the module is in training mode. If
            ``False``, this method would return the inputs directly.

    References:
        1. Larsson et al. (https://arxiv.org/abs/1605.07648)
    """
    if p == 0 or not training:
        return x
    prob = 1 - p
    size = (x.size(0), ) + (1, ) * (x.dim() - 1)
    rand = prob + torch.rand(size, dtype=x.dtype, device=x.device)
    return x / prob * rand.floor()


@NORMS.register(group=['drop', '1d'])
@nncore.bind_getter('p')
class DropPath(nn.Module):
    """
    DropPath operation introducted in [1].

    Args:
        p (float, optional): Probability of the path to be dropped. Default:
            ``0.1``.

    References:
        1. Larsson et al. (https://arxiv.org/abs/1605.07648)
    """

    def __init__(self, p=0.1):
        super(DropPath, self).__init__()
        self._p = p

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self._p)

    def forward(self, x):
        return drop_path(x, self._p, self.training)


@NORMS.register(group=['norm', '1d', '2d', '3d'])
@nncore.bind_getter('eps')
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization introducted in [1].

    Args:
        dims (int): Input feature dimensions.
        eps (float, optional): The term added to the denominator to improve
            numerical stability. Default: ``1e-6``.

    References:
        1. Zhang et al. (https://arxiv.org/abs/1910.07467)
    """

    def __init__(self, dims, eps=1e-6):
        super(RMSNorm, self).__init__()
        self._eps = eps
        self.weight = nn.Parameter(torch.ones(dims))

    def forward(self, x):
        d = x.float()
        d = d * (d.pow(2).mean(dim=-1, keepdim=True) + self._eps).rsqrt()
        x = d.type_as(x) * self.weight
        return x
