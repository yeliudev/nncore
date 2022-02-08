# Copyright (c) Ye Liu. All rights reserved.

import os
import random
from datetime import datetime
from importlib import import_module
from pkgutil import walk_packages

import numpy as np
import torch
import torchvision
from torch.hub import load_state_dict_from_url

import nncore
from nncore.nn import move_to_device
from .comm import broadcast, is_main_process, sync

DATASETS = nncore.Registry('dataset')


def _load_url_dist(url, **kwargs):
    if is_main_process():
        load_state_dict_from_url(url, **kwargs)

    sync()
    state_dict = load_state_dict_from_url(url, **kwargs)

    return state_dict


def _match_keys(keys, cand):
    keys = [k.split('.') for k in keys]
    cand = cand.split('.')

    for key in keys:
        if cand[:len(key)] == key:
            return True

    return False


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
        err_msg.append('Unexpected keys in source state dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if len(missing_keys) > 0:
        err_msg.append('Missing keys in source state dict: {}\n'.format(
            ', '.join(missing_keys)))

    if is_main_process() and len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(
                'error in loading state dict for {}:\n\t{}'.format(
                    module.__class__.__name__, "\n\t".join(err_msg)))
        nncore.log_or_print(err_msg, logger, log_level='WARNING')


def generate_random_seed(sync=True, src=0, group=None):
    """
    Generate a random seed.

    Args:
        sync (bool, optional): Whether to synchronize the random seed among the
            processes in the group in distributed settings. Default: ``True``.
        src (int, optional): The source rank of the process in distributed
            settings. This argument is valid only when ``sync==True``. Default:
            ``0``.
        group (:obj:`dist.ProcessGroup` | None, optional): The process group
            to use in distributed settings. This argument is valid only when
            ``sync==True``. If not specified, the default process group will
            be used. Default: ``None``.

    Returns:
        int: The generated random seed.
    """
    seed = 0
    while len(str(seed)) != 8:
        seed = os.getpid() + int.from_bytes(os.urandom(4), 'big') + int(
            datetime.now().strftime('%f'))
    if sync:
        seed = broadcast(data=seed, src=src, group=group)
    return seed


def set_random_seed(seed=None, benchmark=False, deterministic=False, **kwargs):
    """
    Set random seed for ``random``, ``numpy``, and ``torch`` packages. If
    ``seed`` is not specified, this method will generate and return a new
    random seed.

    Args:
        seed (int | None, optional): The random seed to use. If not specified,
            a new random seed will be generated. Default: ``None``.
        benchmark (bool, optional): Whether to enable benchmark mode. Default:
            ``False``.
        deterministic (bool, optional): Whether to enable deterministic mode.
            Default: ``False``.

    Returns:
        int: The actually used random seed.
    """
    if seed is None:
        seed = generate_random_seed(**kwargs)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

    return seed


def get_checkpoint(file_or_url, map_location=None, **kwargs):
    """
    Get checkpoint from a file or an URL.

    Args:
        file_or_url (str): The filename or URL of the checkpoint.
        map_location (str | None, optional): Same as the :obj:`torch.load`
            interface. Default: ``None``.

    Returns:
        :obj:`OrderedDict` | dict: The loaded checkpoint.
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
                    keys=None,
                    logger=None,
                    **kwargs):
    """
    Load checkpoint from a file or an URL.

    Args:
        model (:obj:`nn.Module`): The module to load checkpoint.
        checkpoint (dict | str): A dict, a filename, an URL or a
            ``torchvision://<model_name>`` str indicating the checkpoint.
        map_location (str | None, optional): Same as the :obj:`torch.load`
            interface. Default: ``None``.
        strict (bool, optional): Whether to allow different params for the
            model and checkpoint. If ``True``, raise an error when the params
            do not match exactly. Default: ``False``.
        keys (list[str] | None, optional): The list of parameter keys to load.
            Default: ``None``.
        logger (:obj:`logging.Logger` | str | None, optional): The logger or
            name of the logger for displaying error messages. Default:
            ``None``.

    Returns:
        :obj:`OrderedDict` | dict: The loaded checkpoint.
    """
    if isinstance(checkpoint, str):
        checkpoint = get_checkpoint(
            checkpoint, map_location=map_location, **kwargs)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        raise RuntimeError('no state dict found in the checkpoint file')

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    if keys is not None:
        state_dict = {
            k: v
            for k, v in state_dict.items() if _match_keys(keys, k)
        }

    _load_state_dict(
        getattr(model, 'module', model),
        state_dict,
        strict=strict,
        logger=logger)

    return checkpoint


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """
    Save checkpoint to a file.

    The checkpoint object will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``, where ``meta`` contains the version of nncore and the time
    info by default.

    Args:
        model (:obj:`nn.Module`): The model whose params are to be saved.
        filename (str): Path to the checkpoint file.
        optimizer (:obj:`optim.Optimizer` | None, optional): The optimizer to
            be saved. Default: ``None``.
        meta (dict | None, optional): The metadata to be saved. Default:
            ``None``.

    Returns:
        dict: The saved checkpoint.
    """
    if meta is None:
        meta = dict()

    meta.update(
        nncore_version=nncore.__version__, create_time=nncore.get_time_str())

    state_dict = getattr(model, 'module', model).state_dict()
    checkpoint = dict(meta=meta, state_dict=state_dict)

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    checkpoint = move_to_device(checkpoint, 'cpu')

    nncore.mkdir(nncore.dir_name(filename))
    torch.save(checkpoint, filename)

    return checkpoint
