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
    Drop paths per sample when applied in main path of residual blocks.

    Args:
        p (float, optional): Probability of the path to be dropped. Default:
            ``0.1``.
        training (bool, optional): Whether the module is in training mode. If
            ``False``, this method would return the inputs directly.
    """
    if p == 0. or not training:
        return x
    keep_prob = 1 - p
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


@NORMS.register(group=['drop', '1d'])
@nncore.bind_getter('p')
class DropPath(nn.Module):
    """
    Drop paths per sample when applied in main path of residual blocks.

    Args:
        p (float, optional): Probability of the path to be dropped. Default:
            ``0.1``.
    """

    def __init__(self, p=0.1):
        super(DropPath, self).__init__()
        self._p = p

    def forward(self, x):
        return drop_path(x, self._p, self.training)
