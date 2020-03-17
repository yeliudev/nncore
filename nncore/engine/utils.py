# Copyright (c) Ye Liu. All rights reserved.

import hashlib
import os.path as osp
import subprocess
from collections import OrderedDict
from functools import partial
from importlib import import_module
from pkgutil import walk_packages
from time import asctime

import torch
import torchvision

import nncore
from .comm import is_main_process, synchronize

_HOOKS = [
    'before_launch', 'after_launch', 'before_stage', 'after_stage',
    'before_train_epoch', 'after_train_epoch', 'before_val_epoch',
    'after_val_epoch', 'before_train_iter', 'after_train_iter',
    'before_val_iter', 'after_val_iter'
]


def bind_hooks(cls):

    def _call_hook(self, name):
        for hook in self._hooks.values():
            getattr(hook, name)(self)

    for hook in _HOOKS:
        setattr(cls, hook, partial(_call_hook, name=hook))

    return cls


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module('torchvision.models.{}'.format(name))
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def load_url_dist(url):
    if is_main_process():
        checkpoint = torch.utils.model_zoo.load_url(url)
    synchronize()
    if not is_main_process():
        checkpoint = torch.utils.model_zoo.load_url(url)
    return checkpoint


def get_checkpoint(filename, map_location=None):
    """
    Get checkpoint from file or url.

    Args:
        filename (str): a filepath or URI
        map_location (str or None, optional): same as :func:`torch.load`

    Returns:
        checkpoint (dict or OrderedDict): the loaded checkpoint. It can be
            either an OrderedDict storing model weights or a dict containing
            other information, which depends on the checkpoint.
    """
    if filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        checkpoint = load_url_dist(model_urls[filename[14:]])
    elif filename.startswith(('https://', 'http://')):
        checkpoint = load_url_dist(filename)
    else:
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """
    Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for `strict` is set to `False` and the message for param
    mismatch will be shown even if strict is False.

    Args:
        module (:obj:`nn.Module`): the module receives the state_dict
        state_dict (OrderedDict): weights
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function.
        logger (:obj:`logging.Logger`, optional): the logger to log the error
            message. If not specified, `print` function will be used.
    """
    unexpected_keys = []
    missing_keys = []
    err = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = dict() if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, err)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [k for k in missing_keys if 'num_batches_tracked' not in k]

    if unexpected_keys:
        err.append('unexpected keys in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))

    if is_main_process() and len(err) > 0:
        err.insert(0, 'The model and loaded state_dict do not match exactly\n')
        err = '\n'.join(err)
        if strict:
            raise RuntimeError(err)
        nncore.log_or_print(err, logger)


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = get_checkpoint(filename, map_location=map_location)

    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'no state_dict found in checkpoint file {}'.format(filename))

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

    The checkpoint object will have 3 fields: meta, state_dict and `optimizer`,
    where `meta` contains nncore version and time info by default.

    Args:
        model (:obj:`nn.Module`): module whose params are to be saved
        filename (str): name of the checkpoint file
        optimizer (:obj:`Optimizer`, optional): the optimizer to be saved
        meta (dict, optional): metadata to be saved
    """
    if meta is None:
        meta = dict()
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(
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
        keys_to_remove (list[str], optional): the keys to be removed from the
            checkpoint
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
