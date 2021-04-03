# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore

NORMS = nncore.Registry('norm')

NORMS.register(nn.BatchNorm1d, name='BN1d', group=['norm', '1d'])
NORMS.register(nn.BatchNorm2d, name=['BN', 'BN2d'], group=['norm', '2d'])
NORMS.register(nn.BatchNorm3d, name='BN3d', group=['norm', '3d'])
NORMS.register(nn.SyncBatchNorm, name='SyncBN', group=['norm', '2d'])
NORMS.register(nn.GroupNorm, name='GN', group=['norm', '2d'])
NORMS.register(nn.LayerNorm, name='LN', group=['norm', '2d'])
NORMS.register(nn.InstanceNorm1d, name='IN1d', group=['norm', '1d'])
NORMS.register(nn.InstanceNorm2d, name=['IN', 'IN2d'], group=['norm', '2d'])
NORMS.register(nn.InstanceNorm3d, name='IN3d', group=['norm', '3d'])
NORMS.register(nn.Dropout, name='Drop1d', group=['drop', '1d'])
NORMS.register(nn.Dropout2d, name=['Drop', 'Drop2d'], group=['drop', '2d'])
NORMS.register(nn.Dropout3d, name='Drop3d', group=['drop', '3d'])
NORMS.register(nn.AlphaDropout, name='ADrop', group=['drop', '1d'])
NORMS.register(nn.FeatureAlphaDropout, name='FADrop', group=['drop', '1d'])


def build_norm_layer(cfg, **kwargs):
    """
    Build a normalization layer from a dict. This method searches for layers
    in :obj:`NORMS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict or str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return nncore.build_object(cfg, [NORMS, nn], **kwargs)
