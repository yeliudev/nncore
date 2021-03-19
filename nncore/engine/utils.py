# Copyright (c) Ye Liu. All rights reserved.

import os
import random
from collections import OrderedDict
from datetime import datetime
from importlib import import_module
from pkgutil import walk_packages
from time import asctime

import numpy as np
import torch
import torchvision

import nncore
from .comm import is_main_process, synchronize


def generate_random_seed(length=8):
    """
    Generate a random seed with at least 8 digits.

    Args:
        length (int, optional): the expected number of digits of the random
            seed. The number must equal or be larger than 8.

    Returns:
        seed (int): the generated random seed
    """
    if length < 8:
        raise ValueError(
            'the number of digits must equal or be larger than 8, but got {}'.
            format(length))
    seed = os.getpid() + int(datetime.now().strftime('%S%f')) + int.from_bytes(
        os.urandom(length - 6), 'big')
    return seed


def set_random_seed(seed=None):
    """
    Set random seed for `random`, `numpy`, and `torch`. If `seed` is None, this
    method will generate and return a new random seed.

    Args:
        seed (int or None, optional): the potential random seed to be used

    Returns:
        seed (int): the actually used random seed
    """
    if seed is None:
        seed = generate_random_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def _load_url_dist(url, **kwargs):
    if is_main_process():
        torch.utils.model_zoo.load_url(url, **kwargs)
    synchronize()
    return torch.utils.model_zoo.load_url(url, **kwargs)


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """
    Load state_dict to a module.

    This method is modified from :meth:`nn.Module.load_state_dict`. Default
    value for `strict` is set to `False` and the message for param mismatch
    will be shown even if strict is `False`.

    Args:
        module (:obj:`nn.Module`): the module receives the state_dict
        state_dict (OrderedDict): weights
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`nn.Module.state_dict` function.
        logger (:obj:`logging.Logger` or str or None, optional): the logger or
            name of the logger for displaying error messages
    """
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


def get_checkpoint(file_or_url, map_location=None, **kwargs):
    """
    Get checkpoint from a file or an URL.

    Args:
        file_or_url (str): a filename or an URL
        map_location (str or None, optional): same as :meth:`torch.load`

    Returns:
        checkpoint (dict or OrderedDict): the loaded checkpoint. It can be
            either an OrderedDict storing model weights or a dict containing
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
        model (Module): the module to load checkpoint
        checkpoint (dict or OrderedDict or str): either a checkpoint object or
            filename or URL or torchvision://<model_name>
        map_location (str or None, optional): same as :meth:`torch.load`
        strict (bool, optional): whether to allow different params for the
            model and checkpoint. If `True`, raise an error when the params do
            not match exactly.
        logger (:obj:`logging.Logger` or str or None, optional): the logger or
            name of the logger for displaying error messages
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

    The checkpoint object will have 3 fields: `meta`, `state_dict`, and
    `optimizer`, where `meta` contains the version of nncore and the time info
    by default.

    Args:
        model (:obj:`nn.Module`): the module whose params are to be saved
        filename (str): name of the checkpoint file
        optimizer (:obj:`Optimizer`, optional): the optimizer to be saved
        meta (dict, optional): the metadata to be saved
    """
    if meta is None:
        meta = dict()
    elif not isinstance(meta, dict):
        raise TypeError("meta must be a dict or None, but got '{}'".format(
            type(meta)))

    meta.update(nncore_version=nncore.__version__, time=asctime())
    nncore.mkdir(nncore.dir_name(filename))

    state_dict = OrderedDict({
        k: v.cpu()
        for k, v in getattr(model, 'module', model).state_dict().items()
    })

    checkpoint = dict(state_dict=state_dict, meta=meta)
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    torch.save(checkpoint, filename)
