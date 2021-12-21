# Copyright (c) Ye Liu. All rights reserved.

import torch

from nncore import Registry, build_object

OPTIMIZERS = Registry('optimizer')


def build_optimizer(cfg, *args, **kwargs):
    """
    Build an optimizer from a dict. This method searches for optimizers in
    :obj:`OPTIMIZERS` first, and then fall back to :obj:`torch.optim`.

    Args:
        cfg (dict): The config of the optimizer.

    Returns:
        :obj:`optim.Optimizer`: The constructed optimizer.
    """
    return build_object(cfg, [OPTIMIZERS, torch.optim], args=args, **kwargs)
