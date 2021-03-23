# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn

import nncore

ACTIVATION_LAYERS = nncore.Registry('activation layer')


class _SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


@ACTIVATION_LAYERS.register()
class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


@ACTIVATION_LAYERS.register()
class EffSwish(nn.Module):

    def forward(self, x):
        return _SwishImplementation.apply(x)


@ACTIVATION_LAYERS.register()
class Clamp(nn.Module):

    def __init__(self, min=-1, max=1):
        super(Clamp, self).__init__()
        self._min = min
        self._max = max

    def forward(self, x):
        return torch.clamp(x, min=self._min, max=self._max)


def build_act_layer(cfg, **kwargs):
    return nncore.build_object(cfg, [ACTIVATION_LAYERS, nn], **kwargs)
