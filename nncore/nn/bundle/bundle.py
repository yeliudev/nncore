# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import torch
import torch.nn as nn

import nncore


class Sequential(nn.Sequential):
    """
    An :obj:`nn.Sequential` class that supports multiple inputs.
    """

    def __init__(self, *args):
        super(nn.Sequential, self).__init__()

        idx = 0
        for arg in args:
            if isinstance(arg, OrderedDict):
                for key, mod in arg.items():
                    if mod is not None:
                        self.add_module(key, mod)
                continue
            elif not isinstance(arg, (list, tuple)):
                arg = [arg]

            for mod in [a for a in arg if a is not None]:
                self.add_module(str(idx), mod)
                idx += 1

    def forward(self, x, **kwargs):
        for mod in self:
            x = mod(x, **kwargs)
        return x


class ModuleList(nn.ModuleList):
    """
    An :obj:`nn.ModuleList` class that supports multiple inputs.
    """

    def __init__(self, *args):
        super(nn.ModuleList, self).__init__()
        self += [a for a in nncore.flatten(args) if a is not None]


class ModuleDict(nn.ModuleDict):
    """
    An :obj:`nn.ModuleDict` class that supports multiple inputs.
    """

    def __init__(self, *args, **kwargs):
        super(ModuleDict, self).__init__()
        for arg in args:
            self.update(arg)
        self.update(kwargs)


class Parameter(nn.Parameter):
    """
    An :obj:`nn.Parameter` class that supports multiple inputs initializes the
    parameters using a scaled normal distribution.
    """

    def __new__(cls, *args, requires_grad=True, **kwargs):
        if torch.is_tensor(args[0]):
            data = args[0]
        elif isinstance(args[0], (list, tuple)):
            data = torch.randn(args[0], **kwargs) / args[0][-1]**0.5
        else:
            data = torch.randn(args, **kwargs) / args[-1]**0.5

        return torch.Tensor._make_subclass(cls, data, requires_grad)
