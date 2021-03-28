# Copyright (c) Ye Liu. All rights reserved.

import hashlib
import os
from itertools import islice

import torch
import torch.nn as nn


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
    During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures.

    Args:
        model (:obj:`nn.Module`): The module whose ``Conv-BN`` structure to be
            fused.

    Returns:
        :obj:`nn.Module`: The module whose ``Conv-BN`` structure has been \
            fused.
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


@torch.no_grad()
def update_bn_stats(model, data_loader, num_iters=200):
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration,
    so the running average can not precisely reflect the actual stats of the
    current model.

    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true
    average of per-batch mean/variance instead of the running average.

    Args:
        model (:obj:`nn.Module`): The model whose bn stats will be recomputed.

            Note that:

            1. This function will not alter the training mode of the given
               model. Users are responsible for setting the layers that needs
               precise-bn to training mode, prior to calling this function.

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

    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    ind = -1
    for ind, inputs in enumerate(islice(data_loader, num_iters)):
        with torch.no_grad():
            model(inputs)

        for i, bn in enumerate(bn_layers):
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
            running_var[i] += (bn.running_var - running_var[i]) / (ind + 1)

    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]


def publish_model(in_file,
                  out_file,
                  keys_to_remove=['optimizer'],
                  hash_type='sha256'):
    """
    Publish a model by removing needless data in a checkpoint and hash the
    output checkpoint file.

    Args:
        in_file (str): Path to the input checkpoint file.
        out_file (str): Path to the output checkpoint file.
        keys_to_remove (list[str], optional): The list of keys to be removed
            from the checkpoint. Default: ``['optimizer']``.
        hash_type (str, optional): Type of the hash algorithm. Currently
            supported algorithms include ``'md5'``, ``'sha1'``, ``'sha224'``,
            ``'sha256'``, ``'sha384'``, ``'sha512'``, ``'blake2b'``,
            ``'blake2s'``, ``'sha3_224'``, ``'sha3_256'``, ``'sha3_384'``,
            ``'sha3_512'``, ``'shake_128'`` and ``'shake_256'``. Default:
            ``'sha256'``.
    """
    checkpoint = torch.load(in_file, map_location='cpu')
    for key in keys_to_remove:
        if key in checkpoint:
            del checkpoint[key]
    torch.save(checkpoint, out_file)

    with open(in_file, 'rb') as f:
        hasher = hashlib.new(hash_type, data=f.read())
        hash_value = hasher.hexdigest()

    final_file = '{}-{}.pth'.format(out_file.rstrip('.pth'), hash_value[:8])
    os.rename(out_file, final_file)
