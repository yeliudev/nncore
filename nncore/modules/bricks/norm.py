# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore

NORM_LAYERS = nncore.Registry('norm layer')

NORM_LAYERS.register(nn.BatchNorm2d, name='BN')
NORM_LAYERS.register(nn.BatchNorm1d, name='BN1d')
NORM_LAYERS.register(nn.BatchNorm2d, name='BN2d')
NORM_LAYERS.register(nn.BatchNorm3d, name='BN3d')
NORM_LAYERS.register(nn.SyncBatchNorm, name='SyncBN')
NORM_LAYERS.register(nn.GroupNorm, name='GN')
NORM_LAYERS.register(nn.LayerNorm, name='LN')
NORM_LAYERS.register(nn.InstanceNorm2d, name='IN')
NORM_LAYERS.register(nn.InstanceNorm1d, name='IN1d')
NORM_LAYERS.register(nn.InstanceNorm2d, name='IN2d')
NORM_LAYERS.register(nn.InstanceNorm3d, name='IN3d')
NORM_LAYERS.register(nn.Dropout, name='Drop')
NORM_LAYERS.register(nn.Dropout, name='Drop1d')
NORM_LAYERS.register(nn.Dropout2d, name='Drop2d')
NORM_LAYERS.register(nn.Dropout3d, name='Drop3d')
NORM_LAYERS.register(nn.AlphaDropout, name='ADrop')
NORM_LAYERS.register(nn.FeatureAlphaDropout, name='FADrop')


def build_norm_layer(cfg, **kwargs):
    return nncore.build_object(cfg, [NORM_LAYERS, nn], **kwargs)
