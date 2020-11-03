# Copyright (c) Ye Liu. All rights reserved.

import hashlib
import os
import os.path as osp
import random
import subprocess
from collections import OrderedDict
from datetime import datetime
from importlib import import_module
from pkgutil import walk_packages
from time import asctime

import numpy as np
import torch
import torch.nn as nn
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
    assert length >= 8, 'the number of digits must equal or be larger than 8'
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


def _load_url_dist(url):
    if is_main_process():
        checkpoint = torch.utils.model_zoo.load_url(url)
    synchronize()
    if not is_main_process():
        checkpoint = torch.utils.model_zoo.load_url(url)
    return checkpoint


def get_checkpoint(file_or_url, map_location=None):
    """
    Get checkpoint from a file or an URL.

    Args:
        file_or_url (str): a filename or an URL
        map_location (str or None, optional): same as :func:`torch.load`

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
            _zoo = import_module('torchvision.models.{}'.format(name))
            if hasattr(_zoo, 'model_urls'):
                _urls = getattr(_zoo, 'model_urls')
                model_urls.update(_urls)
        checkpoint = _load_url_dist(model_urls[file_or_url[14:]])
    elif file_or_url.startswith(('https://', 'http://')):
        checkpoint = _load_url_dist(file_or_url)
    else:
        checkpoint = torch.load(file_or_url, map_location=map_location)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """
    Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for `strict` is set to `False` and the message for param
    mismatch will be shown even if strict is `False`.

    Args:
        module (:obj:`nn.Module`): the module receives the state_dict
        state_dict (OrderedDict): weights
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`torch.nn.Module.state_dict` function.
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


def load_checkpoint(model,
                    checkpoint,
                    map_location=None,
                    strict=False,
                    logger=None):
    """
    Load checkpoint from a file or an URL.

    Args:
        model (Module): the module to load checkpoint
        checkpoint (dict or OrderedDict or str): either a checkpoint object or
            filename or URL or torchvision://<model_name>
        map_location (str, optional): same as :func:`torch.load`
        strict (bool, optional): whether to allow different params for the
            model and checkpoint. If True, raise an error when the params do
            not match exactly.
        logger (:obj:`logging.Logger` or str or None, optional): the logger or
            name of the logger for displaying error messages
    """
    if isinstance(checkpoint, str):
        checkpoint = get_checkpoint(checkpoint, map_location=map_location)
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

    load_state_dict(
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
    nncore.mkdir(osp.dirname(filename))

    state_dict = OrderedDict({
        k: v.cpu()
        for k, v in getattr(model, 'module', model).state_dict().items()
    })

    checkpoint = dict(state_dict=state_dict, meta=meta)
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    torch.save(checkpoint, filename)


def _fuse_conv_bn(conv, bn):
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(model):
    """
    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        model (:obj:`nn.Module`): the module whose conv-bn structure to be
            fused

    Returns:
        fused_model (:obj:`nn.Module`): the module whose conv-bn structure has
            been fused
    """
    last_conv = None
    last_conv_name = None

    for name, child in model.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            model._modules[last_conv_name] = fused_conv
            model._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)

    return model


def publish_model(in_file,
                  out_file,
                  keys_to_remove=['optimizer'],
                  hash_type='sha256'):
    """
    Publish a model by removing needless data in a checkpoint and hash the
    output checkpoint file.

    Args:
        in_file (str): name of the input checkpoint file
        out_file (str): name of the output checkpoint file
        keys_to_remove (list[str], optional): the list of keys to be removed
            from the checkpoint
        hash_type (str, optional): type of the hash algorithm. Currently
            supported algorithms include `md5`, `sha1`, `sha224`, `sha256`,
            `sha384`, `sha512`, `blake2b`, `blake2s`, `sha3_224`, `sha3_256`,
            `sha3_384`, `sha3_512`, `shake_128`, and `shake_256`.
    """
    checkpoint = torch.load(in_file, map_location='cpu')
    for key in keys_to_remove:
        if key in checkpoint:
            del checkpoint[key]
    torch.save(checkpoint, out_file)

    with open(in_file, 'rb') as f:
        hasher = hashlib.new(hash_type, data=f.read())
        hash_value = hasher.hexdigest()

    final_file = out_file[:-4] + '-{}.pth'.format(hash_value[:8])
    subprocess.Popen(['mv', out_file, final_file])
