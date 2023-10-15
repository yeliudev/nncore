# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch.nn as nn

from nncore import Registry

INITIALIZERS = Registry('initializer')


@INITIALIZERS.register(name='constant')
def constant_init_(module, value=1, bias=0):
    """
    Initialize a module using a constant.

    Args:
        module (:obj:`nn.Module`): The module to be initialized.
        value (int, optional): The value to be filled. Default: ``1``.
        bias (int, optional): The bias of the module. Default: ``0``.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, value)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@INITIALIZERS.register(name='normal')
def normal_init_(module, mean=0, std=1, bias=0):
    """
    Initialize a module using normal distribution.

    Args:
        module (:obj:`nn.Module`): The module to be initialized.
        mean (int, optional): Mean of the distribution. Default: ``0``.
        std (int, optional): Standard deviation of the distribution. Default:
            ``1``.
        bias (int, optional): The bias of the module. Default: ``0``.
    """
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@INITIALIZERS.register(name='uniform')
def uniform_init_(module, a=0, b=1, bias=0):
    """
    Initialize a module using uniform distribution.

    Args:
        module (:obj:`nn.Module`): The module to be initialized.
        a (int, optional): Lower bound of the distribution. Default: ``0``.
        b (int, optional): Upper bound of the distribution. Default: ``1``.
        bias (int, optional): The bias of the module. Default: ``0``.
    """
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@INITIALIZERS.register(name='xavier')
def xavier_init_(module, gain=1, bias=0, distribution='normal'):
    """
    Initialize a module using the method introduced in [1].

    Args:
        module (:obj:`nn.Module`): The module to be initialized.
        gain (int, optional): The scaling factor. Default: ``1``.
        bias (int, optional): The bias of the module. Default: ``0``.
        distribution (str, optional): The type of distribution to use.
            Expected values include ``normal`` and ``uniform``. Default:
            ``'normal'``.

    References:
        1. Glorot et al. (http://proceedings.mlr.press/v9/glorot10a)
    """
    assert distribution in ('normal', 'uniform')
    if distribution == 'normal':
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


@INITIALIZERS.register(name='kaiming')
def kaiming_init_(module,
                  a=0,
                  mode='fan_in',
                  nonlinearity='leaky_relu',
                  bias=0,
                  distribution='normal'):
    """
    Initialize a module using the method introduced in [1].

    Args:
        module (:obj:`nn.Module`): The module to be initialized.
        a (int, optional): The negative slope of ``LeakyReLU``. Default: ``0``.
        mode (str, optional): The direction of pass whose magnitude of the
            variance of the weights are preserved. Expected values include
            ``'fan_in'`` and ``'fan_out'``. Default: ``'fan_in'``.
        nonlinearity (str, optional): The nonlinearity after the parameterized
            layers. The expected values are ``'relu'`` and ``'leaky_relu'``.
            Default: ``'leaky_relu'``.
        bias (int, optional): The bias of the module. Default: ``0``.
        distribution (str, optional): The type of distribution to use.
            Expected values include ``normal`` and ``uniform``. Default:
            ``'normal'``.

    References:
        1. He et al. (https://arxiv.org/abs/1502.01852)
    """
    assert distribution in ('normal', 'uniform')
    if distribution == 'normal':
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def init_module_(module, init_cfg):
    """
    Initialize a module using the specified method.

    Args:
        module (:obj:`nn.Module`): The module to be initialized.
        init_cfg (dict | str | None): The initialization method and configs.
            If ``None``, the module will not be initialized.
    """
    if init_cfg is not None:
        _cfg = init_cfg.copy()
        method = _cfg.pop('type')
        INITIALIZERS.get(method)(module, **_cfg)
