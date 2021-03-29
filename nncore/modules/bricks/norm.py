# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore

NORM_LAYERS = nncore.Registry('norm layer')

NORM_LAYERS.register(nn.BatchNorm2d, name='BN', group='2d')
NORM_LAYERS.register(nn.BatchNorm1d, name='BN1d', group='1d')
NORM_LAYERS.register(nn.BatchNorm2d, name='BN2d', group='2d')
NORM_LAYERS.register(nn.BatchNorm3d, name='BN3d', group='3d')
NORM_LAYERS.register(nn.SyncBatchNorm, name='SyncBN', group='2d')
NORM_LAYERS.register(nn.GroupNorm, name='GN', group='2d')
NORM_LAYERS.register(nn.LayerNorm, name='LN', group='2d')
NORM_LAYERS.register(nn.InstanceNorm2d, name='IN', group='2d')
NORM_LAYERS.register(nn.InstanceNorm1d, name='IN1d', group='1d')
NORM_LAYERS.register(nn.InstanceNorm2d, name='IN2d', group='2d')
NORM_LAYERS.register(nn.InstanceNorm3d, name='IN3d', group='3d')
NORM_LAYERS.register(nn.Dropout, name='Drop', group='1d')
NORM_LAYERS.register(nn.Dropout, name='Drop1d', group='1d')
NORM_LAYERS.register(nn.Dropout2d, name='Drop2d', group='2d')
NORM_LAYERS.register(nn.Dropout3d, name='Drop3d', group='3d')
NORM_LAYERS.register(nn.AlphaDropout, name='ADrop', group='1d')
NORM_LAYERS.register(nn.FeatureAlphaDropout, name='FADrop', group='1d')


def build_norm_layer(cfg, **kwargs):
    return nncore.build_object(cfg, [NORM_LAYERS, nn], **kwargs)
