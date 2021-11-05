# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

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
