# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

import nncore

CONVS = nncore.Registry('conv')

CONVS.register(nn.Conv1d, group='1d')
CONVS.register(nn.Conv2d, name=['Conv', 'Conv2d'], group='2d')
CONVS.register(nn.Conv3d, group='3d')
CONVS.register(nn.ConvTranspose1d, name='ConvT1d', group='1d')
CONVS.register(nn.ConvTranspose2d, name=['ConvT', 'ConvT2d'], group='2d')
CONVS.register(nn.ConvTranspose3d, name='ConvT3d', group='3d')
CONVS.register(nn.LazyConv1d, name='LConv1d', group='1d')
CONVS.register(nn.LazyConv2d, name=['LConv', 'LConv2d'], group='2d')
CONVS.register(nn.LazyConv3d, name='LConv3d', group='3d')
CONVS.register(nn.LazyConvTranspose1d, name='LConvT1d', group='1d')
CONVS.register(nn.LazyConvTranspose2d, name=['LConvT', 'LConvT2d'], group='2d')
CONVS.register(nn.LazyConvTranspose3d, name='LConvT3d', group='3d')


def build_conv_layer(cfg, **kwargs):
    """
    Build a convolution layer from a dict. This method searches for layers in
    :obj:`CONVS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict or str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return nncore.build_object(cfg, [CONVS, nn], **kwargs)
