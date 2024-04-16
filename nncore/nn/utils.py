# Copyright (c) Ye Liu. Licensed under the MIT License.

import hashlib
from collections import OrderedDict
from itertools import islice

import torch
import torch.nn as nn

import nncore


def move_to_device(data, device='cpu'):
    """
    Recursively move a tensor or a collection of tensors to the specific
    device.

    Args:
        data (dict | list | :obj:`torch.Tensor`): The tensor or collection of
            tensors to be moved.
        device (:obj:`torch.device` | str, optional): The destination device.
            Default: ``'cpu'``.

    Returns:
        dict | list | :obj:`torch.Tensor`: The moved tensor or collection of \
            tensors.
    """
    if isinstance(data, dict):
        return data.__class__({
            k: move_to_device(v, device=device)
            for k, v in data.items()
        })
    elif isinstance(data, (list, tuple)):
        return type(data)([move_to_device(d, device=device) for d in data])
    elif torch.is_tensor(data):
        return data.to(device)
    else:
        return data


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


def update_bn_stats_(model, data_loader, num_iters=200, **kwargs):
    """
    Recompute and update the BN stats to make them more precise. During
    training, both BN stats and the weight are changing after every iteration,
    so the running average can not precisely reflect the actual stats of the
    current model. In this function, the BN stats are recomputed with fixed
    weights to make the running average more precise. Specifically, it
    computes the true average of per-batch mean/variance instead of the
    running average.

    Args:
        model (:obj:`nn.Module`): The model whose BN stats will be recomputed.
            Note that:

            1. This function will not alter the training mode of the given
               model. Users are responsible for setting the layers that needs
               Precise-BN to training mode, prior to calling this function.
            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations. This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers or backup the states.

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
        with torch.no_grad():
            model(inputs, **kwargs)

        for i, bn in enumerate(bn_layers):
            bn_rm[i] += (bn.running_mean - bn_rm[i]) / (ind + 1)
            bn_rv[i] += (bn.running_var - bn_rv[i]) / (ind + 1)

        prog_bar.update()

    for i, bn in enumerate(bn_layers):
        bn.running_mean = bn_rm[i]
        bn.running_var = bn_rv[i]
        bn.momentum = bn_mo[i]


def publish_model(checkpoint,
                  out='model.pth',
                  keys_to_keep=['state_dict'],
                  device='cpu',
                  meta=None,
                  hash_type='sha256',
                  hash_len=8):
    """
    Publish a model by removing needless data in the checkpoint, moving the
    weights to the specified device, and hashing the output model file.

    Args:
        checkpoint (dict | str): The checkpoint or path to the checkpoint.
        out (str, optional): Path to the output checkpoint file. Default:
            ``'model.pth'``.
        keys_to_keep (list[str], optional): The list of keys to be kept from
            the checkpoint. Default: ``['state_dict']``.
        device (:obj:`torch.device` | str): The destination device. Default:
            ``'cpu'``.
        meta (dict | None, optional): The meta data to be saved. Note that the
            key ``nncore_version`` and ``create_time`` are reserved by the
            method. Default: ``None``.
        hash_type (str | None, optional): Type of the hash algorithm. Currently
            supported algorithms include ``'md5'``, ``'sha1'``, ``'sha224'``,
            ``'sha256'``, ``'sha384'``, ``'sha512'``, ``'blake2b'``,
            ``'blake2s'``, ``'sha3_224'``, ``'sha3_256'``, ``'sha3_384'``,
            ``'sha3_512'``, ``'shake_128'``, and ``'shake_256'``. Default:
            ``'sha256'``.
        hash_len (int, optional): Length of the hash value. Default: ``8``.
    """
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    elif not isinstance(checkpoint, dict):
        raise TypeError(
            "checkpoint must be a dict or str, but got '{}'".format(
                type(checkpoint)))

    model = {k: v for k, v in checkpoint.items() if k in keys_to_keep}

    _meta = model.get('meta', dict())
    _meta.update(
        nncore_version=nncore.__version__,
        create_time=nncore.get_time_str(),
        **meta or dict())
    model['meta'] = _meta

    model = move_to_device(model, device=device)
    torch.save(model, out)

    if hash_type is not None:
        with open(out, 'rb') as f:
            hasher = hashlib.new(hash_type, data=f.read())
            hash_value = hasher.hexdigest()[:hash_len]

        name, ext = nncore.split_ext(out)
        hashed = '{}-{}.{}'.format(name, hash_value, ext).rstrip('.')
        nncore.rename(out, hashed)


def model_soup(model1, model2, out='model.pth', device='cpu'):
    """
    Combine two models by calculating the element-wise average of their weight
    matrices (i.e. cooking model soups [1]). The output model is expected to
    have better performance compaired with the original ones.

    Args:
        model1 (dict | str): The checkpoint or path to the checkpoint of the
            first model.
        model2 (dict | str): The checkpoint or path to the checkpoint of the
            second model.
        out (str, optional): Path to the output checkpoint file. Default:
            ``'model.pth'``.
        device (:obj:`torch.device` | str): The destination device. Default:
            ``'cpu'``.

    References:
        1. Wortsman et al. (https://arxiv.org/abs/2203.05482)
    """
    if isinstance(model1, str):
        model1 = torch.load(model1, map_location=device)
    elif not isinstance(model1, dict):
        raise TypeError("model1 must be a dict or str, but got '{}'".format(
            type(model1)))

    if isinstance(model2, str):
        model2 = torch.load(model2, map_location=device)
    elif not isinstance(model2, dict):
        raise TypeError("model2 must be a dict or str, but got '{}'".format(
            type(model2)))

    model1 = model1['state_dict']
    model2 = model2['state_dict']
    assert model1.keys() == model2.keys()

    state_dict = OrderedDict()
    for key in model1.keys():
        state_dict[key] = (model1[key] + model2[key]) / 2

    model = dict(
        state_dict=state_dict,
        meta=dict(
            nncore_version=nncore.__version__,
            create_time=nncore.get_time_str()))

    model = move_to_device(model, device=device)
    torch.save(model, out)
