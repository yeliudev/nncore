# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch.nn as nn

from nncore import Registry, build_object
from nncore.parallel import NNDataParallel, NNDistributedDataParallel
from .bundle import ModuleList, Sequential

MODELS = Registry('model')
ACTIVATIONS = Registry('activation', parent=MODELS)
CONVS = Registry('conv', parent=MODELS)
MESSAGE_PASSINGS = Registry('message passing', parent=MODELS)
NORMS = Registry('norm', parent=MODELS)
LOSSES = Registry('loss', parent=MODELS)
MODULES = Registry('module', parent=MODELS)


def build_model(cfg,
                *args,
                bundler='sequential',
                dist=None,
                dp_cfg=None,
                **kwargs):
    """
    Build a general model from a dict or str. This method searches for modules
    in :obj:`MODELS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the model.
        bundler (str | None, optional): The type of bundler for multiple
            models. Expected values include ``'sequential'`` and
            ``'modulelist'``. Default: ``'sequential'``.
        dist (bool | None, optional): Whether the model is distributed. If not
            specified, the model will not be wrapped. Default: ``None``.
        dp_cfg (dict | None, optional): The config for :obj:`NNDataParallel` or
            :obj:`NNDataParallel`. Default: ``None``.

    Returns:
        :obj:`nn.Module`: The constructed model.
    """
    assert bundler in ('sequential', 'modulelist')

    model = build_object(cfg, [MODELS, nn], args=args, **kwargs)

    if isinstance(model, (list, tuple)):
        model = [m for m in model if m is not None]
        if bundler == 'sequential':
            model = Sequential(model) if len(model) > 1 else model[0]
    elif model is None:
        return

    if bundler == 'modulelist':
        model = ModuleList(model)

    if dp_cfg is None:
        dp_cfg = dict()

    if dist:
        model = NNDistributedDataParallel(model, **dp_cfg)
    elif dist is not None:
        model = NNDataParallel(model, **dp_cfg)

    return model


def build_act_layer(cfg, *args, **kwargs):
    """
    Build an activation layer from a dict or str. This method searches for
    layers in :obj:`ACTIVATIONS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return build_object(cfg, [ACTIVATIONS, nn], args=args, **kwargs)


def build_conv_layer(cfg, *args, **kwargs):
    """
    Build a convolution layer from a dict or str. This method searches for
    layers in :obj:`CONVS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return build_object(cfg, [CONVS, nn], args=args, **kwargs)


def build_msg_pass_layer(cfg, *args, **kwargs):
    """
    Build a message passing layer from a dict or str. This method searches for
    layers in :obj:`MESSAGE_PASSINGS` first, and then fall back to
    :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the layer.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    return build_object(cfg, [MESSAGE_PASSINGS, nn], args=args, **kwargs)


def build_norm_layer(cfg, *args, dims=None, **kwargs):
    """
    Build a normalization layer from a dict or str. This method searches for
    layers in :obj:`NORMS` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the layer.
        dims (int | None, optional): The input dimensions of the layer.
            Default: ``None``.

    Returns:
        :obj:`nn.Module`: The constructed layer.
    """
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    elif not isinstance(cfg, dict):
        return cfg

    _cfg = cfg.copy()

    if dims is not None and _cfg['type'] not in NORMS.group('drop'):
        if _cfg['type'] == 'RMSNorm':
            key = 'dims'
        elif _cfg['type'] == 'LN':
            key = 'normalized_shape'
        elif _cfg['type'] == 'GN':
            key = 'num_channels'
        else:
            key = 'num_features'

        _cfg.setdefault(key, dims)

    return build_object(_cfg, [NORMS, nn], args=args, **kwargs)


def build_loss(cfg, *args, **kwargs):
    """
    Build a loss module from a dict or str. This method searches for modules in
    :obj:`LOSSES` first, and then fall back to :obj:`torch.nn`.

    Args:
        cfg (dict | str): The config or name of the module.

    Returns:
        :obj:`nn.Module`: The constructed module.
    """
    return build_object(cfg, [LOSSES, nn], args=args, **kwargs)
