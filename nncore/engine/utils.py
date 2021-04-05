# Copyright (c) Ye Liu. All rights reserved.

import os
import random
from datetime import datetime

import numpy as np
import torch


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
