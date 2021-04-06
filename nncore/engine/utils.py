# Copyright (c) Ye Liu. All rights reserved.

import os
import random
from collections import OrderedDict
from datetime import datetime
from importlib import import_module
from pkgutil import walk_packages

import numpy as np
import torch
import torchvision

import nncore
from nncore.nn import move_to_device
from .comm import is_main_process, synchronize


def _load_url_dist(url, **kwargs):
    if is_main_process():
        torch.utils.model_zoo.load_url(url, **kwargs)
    synchronize()
    return torch.utils.model_zoo.load_url(url, **kwargs)


def _load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def _load(module, prefix=''):
        local_metadata = dict() if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                _load(child, prefix + name + '.')

    _load(module)
    _load = None

    if len(unexpected_keys) > 0:
        err_msg.append('unexpected keys in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if len(missing_keys) > 0:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))

    if is_main_process() and len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state_dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    module.__class__.__name__, "\n\t".join(err_msg)))
        nncore.log_or_print(err_msg, logger, log_level='WARNING')


def generate_random_seed(length=8):
    """
    Generate a random seed.

    Args:
        length (int, optional): The expected number of digits of the random
            seed. The number must equal or be larger than 8. Default: ``8``.

    Returns:
        int: The generated random seed.
    """
    if length < 8:
        raise ValueError(
            'the number of digits must equal or be larger than 8, but got {}'.
            format(length))
    seed = os.getpid() + int(datetime.now().strftime('%S%f')) + int.from_bytes(
        os.urandom(length - 6), 'big')
    return seed


def set_random_seed(seed=None, deterministic=False, benchmark=False):
    """
    Set random seed for ``random``, ``numpy`` and ``torch`` packages. If
    ``seed`` is not specified, this method will generate and return a new
    random seed.

    Args:
        seed (int or None, optional): The potential random seed to use.
            If not specified, a new random seed will be generated. Default:
            ``None``.
        deterministic (bool, optional): Whether to enable deterministic mode.
            Default: ``False``.
        benchmark (bool, optional): Whether to enable benchmark mode. Default:
            ``False``.

    Returns:
        int: The actually used random seed.
    """
    if seed is None:
        seed = generate_random_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    return seed


def get_checkpoint(file_or_url, map_location=None, **kwargs):
    """
    Get checkpoint from a file or an URL.

    Args:
        file_or_url (str): The filename or URL of the checkpoint.
        map_location (str or None, optional): Same as the :obj:`torch.load`
            interface. Default: ``None``.

    Returns:
        :obj:`OrderedDict` or dict: The loaded checkpoint. It can be either \
            an :obj:`OrderedDict` storing model weights or a dict containing \
            other information, which depends on the checkpoint.
    """
    if file_or_url.startswith('torchvision://'):
        model_urls = dict()
        for _, name, ispkg in walk_packages(torchvision.models.__path__):
            if ispkg:
                continue
            mod = import_module('torchvision.models.{}'.format(name))
            if hasattr(mod, 'model_urls'):
                urls = getattr(mod, 'model_urls')
                model_urls.update(urls)
        checkpoint = _load_url_dist(model_urls[file_or_url[14:]], **kwargs)
    elif file_or_url.startswith(('http://', 'https://')):
        checkpoint = _load_url_dist(file_or_url, **kwargs)
    else:
        checkpoint = torch.load(file_or_url, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    checkpoint,
                    map_location=None,
                    strict=False,
                    logger=None,
                    **kwargs):
    """
    Load checkpoint from a file or an URL.

    Args:
        model (:obj:`nn.Module`): The module to load checkpoint.
        checkpoint (dict or str): A dict, a filename, an URL or a
            ``torchvision://<model_name>`` str indicating the checkpoint.
        map_location (str or None, optional): Same as the :obj:`torch.load`
            interface.
        strict (bool, optional): Whether to allow different params for the
            model and checkpoint. If ``True``, raise an error when the params
            do not match exactly. Default: ``False``.
        logger (:obj:`logging.Logger` or str or None, optional): The logger or
            name of the logger for displaying error messages. Default:
            ``None``.
    """
    if isinstance(checkpoint, str):
        checkpoint = get_checkpoint(
            checkpoint, map_location=map_location, **kwargs)
    elif not isinstance(checkpoint, dict):
        raise TypeError(
            "checkpoint must be a dict or str, but got '{}'".format(
                type(checkpoint)))

    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('no state_dict found in the checkpoint file')

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    _load_state_dict(
        getattr(model, 'module', model),
        state_dict,
        strict=strict,
        logger=logger)


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """
    Save checkpoint to a file.

    The checkpoint object will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``, where ``meta`` contains the version of nncore and the time
    info by default.

    Args:
        model (:obj:`nn.Module`): The model whose params are to be saved.
        filename (str): Path to the checkpoint file.
        optimizer (:obj:`optim.Optimizer`, optional): The optimizer to be
            saved. Default: ``None``.
        meta (dict, optional): The metadata to be saved. Default: ``None``.
    """
    if meta is None:
        meta = dict()
    elif not isinstance(meta, dict):
        raise TypeError("meta must be a dict or None, but got '{}'".format(
            type(meta)))

    meta.update(
        nncore_version=nncore.__version__, create_time=nncore.get_time_str())
    nncore.mkdir(nncore.dir_name(nncore.abs_path(filename)))

    checkpoint = dict(
        meta=meta, state_dict=getattr(model, 'module', model).state_dict())

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    checkpoint = move_to_device(checkpoint, 'cpu')
    torch.save(checkpoint, filename)
