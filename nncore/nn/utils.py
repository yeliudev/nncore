# Copyright (c) Ye Liu. All rights reserved.

import hashlib
from collections import OrderedDict
from importlib import import_module
from itertools import islice
from pkgutil import walk_packages

import torch
import torch.nn as nn
import torchvision

import nncore
from nncore.engine import comm


def _load_url_dist(url, **kwargs):
    if comm.is_main_process():
        torch.utils.model_zoo.load_url(url, **kwargs)
    comm.synchronize()
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

    if comm.is_main_process() and len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state_dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    module.__class__.__name__, "\n\t".join(err_msg)))
        nncore.log_or_print(err_msg, logger, log_level='WARNING')


def move_to_device(data, device):
    """
    Recursively move a tensor or a collection of tensors to the specific
    device.

    Args:
        data (dict or list or :obj:`torch.Tensor`): The tensor or collection
            of tensors to move.
        device (:obj:`torch.device` or str): The destination device.

    Returns:
        dict or list or :obj:`torch.Tensor`: The move tensor or \
            collection of tensors.
    """
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([move_to_device(d, device) for d in data])
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


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
            ``torchvision://<model_name>`` string indicating the checkpoint.
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

    state_dict = OrderedDict({
        k: v.cpu()
        for k, v in getattr(model, 'module', model).state_dict().items()
    })

    checkpoint = dict(state_dict=state_dict, meta=meta)
    if optimizer is not None:
        checkpoint['optimizer'] = move_to_device(optimizer, 'cpu')

    torch.save(checkpoint, filename)


def fuse_bn_(model):
    """
    During inference, the functionary of batch norm layers is turned off but
    only the mean and var are used, which exposes the chance to fuse it with
    the preceding convolution or linear layers to simplify the network
    structure and save computations.

    Args:
        model (:obj:`nn.Module`): The model whose ``Conv-BN`` and ``Linear-BN``
            structure to be fused.

    Returns:
        :obj:`nn.Module`: The model whose ``Conv-BN`` and ``Linear-BN`` \
            structure has been fused.
    """
    last_layer_type = last_layer_name = last_layer = None

    for name, layer in model.named_children():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            if last_layer is None:
                continue

            last_layer = last_layer.clone()
            mo_w, mo_b = last_layer.weight, last_layer.bias
            bn_rm, bn_rv = layer.running_mean, layer.running_var
            bn_w, bn_b, bn_eps = layer.weight, layer.bias, layer.eps

            if mo_b is None:
                mo_b = torch.zeros_like(bn_rm)
            if bn_w is None:
                bn_w = torch.ones_like(bn_rm)
            if bn_b is None:
                bn_b = torch.zeros_like(bn_rm)

            if last_layer_type == 'conv':
                bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
                last_layer.weight = mo_w * (
                    bn_w * bn_var_rsqrt).reshape([-1] + [1] * (mo_w.dim() - 1))
                last_layer.bias = (mo_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
            else:
                bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
                last_layer.weight = mo_w * bn_scale.unsqueeze(-1)
                last_layer.bias = (mo_b - bn_rm) * bn_scale + bn_b

            model._modules[last_layer_name] = last_layer
            model._modules[name] = nn.Identity()

            last_layer = None
        elif isinstance(layer, nn.Conv2d):
            last_layer_type = 'conv'
            last_layer_name = name
            last_layer = layer
        elif isinstance(layer, nn.Linear):
            last_layer_type = 'linear'
            last_layer_name = name
            last_layer = layer
        else:
            fuse_bn_(layer)


@torch.no_grad()
def update_bn_stats_(model, data_loader, num_iters=200, **kwargs):
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration,
    so the running average can not precisely reflect the actual stats of the
    current model.

    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true
    average of per-batch mean/variance instead of the running average.

    Args:
        model (:obj:`nn.Module`): The model whose BN stats will be recomputed.

            Note that:

            1. This function will not alter the training mode of the given
               model. Users are responsible for setting the layers that needs
               Precise-BN to training mode, prior to calling this function.

            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations.  This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers, or backup the states.

        data_loader (iterator): The data loader to use.
        num_iters (int, optional): Number of iterations to compute the stats.
            Default: ``200``.
    """
    assert len(data_loader) >= num_iters

    bn_layers = [
        m for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training
    ]

    if len(bn_layers) == 0:
        return

    bn_mo = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    bn_rm = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    bn_rv = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    prog_bar = nncore.ProgressBar(num_tasks=num_iters)
    for ind, inputs in enumerate(islice(data_loader, num_iters)):
        model(inputs, **kwargs)
        for i, bn in enumerate(bn_layers):
            bn_rm[i] += (bn.running_mean - bn_rm[i]) / (ind + 1)
            bn_rv[i] += (bn.running_var - bn_rv[i]) / (ind + 1)
        prog_bar.update()

    for i, bn in enumerate(bn_layers):
        bn.running_mean = bn_rm[i]
        bn.running_var = bn_rv[i]
        bn.momentum = bn_mo[i]


def publish_model(in_file,
                  out_file,
                  keys_to_remove=['optimizer'],
                  hash_type='sha256'):
    """
    Publish a model by removing needless data in a checkpoint and hash the
    output checkpoint file.

    Args:
        in_file (str): Path to the input checkpoint file.
        out_file (str): Path to the output checkpoint file. It is expected to
            end with ``'.pth'``.
        keys_to_remove (list[str], optional): The list of keys to be removed
            from the checkpoint. Default: ``['optimizer']``.
        hash_type (str, optional): Type of the hash algorithm. Currently
            supported algorithms include ``'md5'``, ``'sha1'``, ``'sha224'``,
            ``'sha256'``, ``'sha384'``, ``'sha512'``, ``'blake2b'``,
            ``'blake2s'``, ``'sha3_224'``, ``'sha3_256'``, ``'sha3_384'``,
            ``'sha3_512'``, ``'shake_128'`` and ``'shake_256'``. Default:
            ``'sha256'``.
    """
    assert out_file.endswith('.pth')
    nncore.file_exist(in_file, raise_error=True)

    checkpoint = torch.load(in_file, map_location='cpu')
    for key in keys_to_remove:
        if key in checkpoint:
            del checkpoint[key]
    torch.save(checkpoint, out_file)

    with open(out_file, 'rb') as f:
        hasher = hashlib.new(hash_type, data=f.read())
        hash_value = hasher.hexdigest()

    hashed_file = '{}-{}.pth'.format(out_file[:-4], hash_value[:8])
    nncore.rename(out_file, hashed_file)
