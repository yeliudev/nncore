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
        for arg in args:
            if isinstance(arg, OrderedDict):
                for key, mod in arg.items():
                    self.add_module(key, mod)
                continue
            elif not isinstance(arg, (list, tuple)):
                arg = [arg]

            for idx, mod in enumerate(args):
                self.add_module(str(idx), mod)


class ModuleList(nn.ModuleList):
    """
    An :obj:`nn.ModuleList` class that supports multiple inputs.
    """

    def __init__(self, *args):
        super(nn.ModuleList, self).__init__()
        mods = [m if isinstance(m, (list, tuple)) else [m] for m in args]
        self += nncore.concat_list(mods)


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
    An :obj:`nn.Parameter` class that supports multiple inputs.
    """

    def __new__(cls, *args, requires_grad=True, **kwargs):
        if torch.is_tensor(args[0]):
            data = args[0]
        elif isinstance(args[0], (list, tuple)):
            data = torch.empty(args[0], **kwargs)
        else:
            data = torch.empty(args, **kwargs)

        return torch.Tensor._make_subclass(cls, data, requires_grad)
