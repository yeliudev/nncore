# Copyright (c) Ye Liu. All rights reserved.

import torch.nn as nn

from nncore import Registry, build_object
from nncore.parallel import NNDataParallel, NNDistributedDataParallel

MODELS = Registry('block')
ACTIVATIONS = Registry('activation', parent=MODELS)
CONVS = Registry('conv', parent=MODELS)
MESSAGE_PASSINGS = Registry('message passing', parent=MODELS)
NORMS = Registry('norm', parent=MODELS)
LOSSES = Registry('loss', parent=MODELS)
MODULES = Registry('module', parent=MODELS)


def build_model(cfg, *args, bundle_type=None, wrap_type=None, **kwargs):
    """
    Build a general model from a dict. This method searches for modules in
    :obj:`MODELS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the model.
        bundle_type (str | None, optional): The type of bundler for multiple
            models. Expected values include ``'sequential'``, ``'modulelist'``,
            and ``None``. Default: ``None``.
        wrap_type (str | None, optional): The type of wrapper for the model.
            Expected values include ``'dp'``, ``'ddp'``, and ``None``. Default:
            ``None``.

    Returns:
        :obj:`nn.Module`: The constructed model.
    """
    assert bundle_type in ('sequential', 'modulelist', None)
    assert wrap_type in ('dp', 'ddp', None)

    obj = build_object(cfg, [MODELS, nn], args=args, **kwargs)

    if isinstance(cfg, (list, tuple)):
        if bundle_type == 'sequential':
            obj = nn.Sequential(*obj)
        elif bundle_type == 'modulelist':
            obj = nn.ModuleList(obj)

    if wrap_type == 'dp':
        obj = NNDataParallel(obj)
    elif wrap_type == 'ddp':
        obj = NNDistributedDataParallel(obj)

    return obj


def build_act_layer(cfg, *args, **kwargs):
    """
    Build an activation layer from a dict. This method searches for layers in
    :obj:`ACTIVATIONS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return build_object(cfg, [ACTIVATIONS, nn], args=args, **kwargs)


def build_conv_layer(cfg, *args, **kwargs):
    """
    Build a convolution layer from a dict. This method searches for layers in
    :obj:`CONVS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return build_object(cfg, [CONVS, nn], args=args, **kwargs)


def build_msg_pass_layer(cfg, *args, **kwargs):
    """
    Build a message passing layer from a dict. This method searches for layers
    in :obj:`MESSAGE_PASSINGS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return build_object(cfg, [MESSAGE_PASSINGS, nn], args=args, **kwargs)


def build_norm_layer(cfg, *args, dims=None, **kwargs):
    """
    Build a normalization layer from a dict. This method searches for layers
    in :obj:`NORMS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the layer.
        dims (int | None, optional): The input dimensions of the layer.
            Default: ``None``.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    if isinstance(cfg, str):
        cfg = dict(type=cfg)

    if dims is not None and cfg['type'] not in NORMS.group('drop'):
        key = 'normalized_shape' if cfg['type'] == 'LN' else 'num_features'
        cfg.setdefault(key, dims)

    return build_object(cfg, [NORMS, nn], args=args, **kwargs)


def build_loss(cfg, *args, **kwargs):
    """
    Build a loss module from a dict. This method searches for modules in
    :obj:`LOSSES` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the module.

    Returns:
        :obj:`nn.Module`: The constructed module.
    """
    return build_object(cfg, [LOSSES, nn], args=args, **kwargs)
