# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch
import torch.nn as nn

from ..builder import MODELS


@MODELS.register()
class Scale(nn.Module):
    """
    Learnable scale layer.

    Args:
        init_value (float, optional): The initial scale value. Default: ``1``.
    """

    def __init__(self, init_value=1):
        super(Scale, self).__init__()
        self._scale = nn.Parameter(torch.Tensor([init_value]))

    def forward(self, x):
        return x * self._scale


@MODELS.register()
class Gate(nn.Module):
    """
    Learnable gate layer.

    Args:
        init_value (float, optional): The initial gate value. Default: ``1``.
    """

    def __init__(self, init_value=1):
        super(Gate, self).__init__()
        self._gate = nn.Parameter(torch.Tensor([init_value]))

    def forward(self, a, b):
        return a * self._gate + b * (1 - self._gate)
