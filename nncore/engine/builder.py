# Copyright (c) Ye Liu. All rights reserved.

import random
from functools import partial

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from nncore import Registry
from nncore.dataset import build_dataset
from nncore.parallel import collate
from .comm import get_dist_info, is_distributed

HOOKS = Registry('hook')


def _init_fn(worker_id, num_workers, rank, seed):
    worker_seed = seed + worker_id + rank * num_workers
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(cfg, seed=None, dist=None, group=None, **kwargs):
    """
    Build a data loader from a dict. The dataset should be registered in
    :obj:`DATASETS`.

    Args:
        cfg (dict): The config of the dataset.
        seed (int | None, optional): The random seed to use. Default: ``None``.
        dist (bool | None, optional): Whether the data loader is distributed.
            If not specified, this method will determine it automatically.
            Default: ``None``.
        group (:obj:`dist.ProcessGroup` | None, optional): The process group
            to use. If not specified, the default process group will be used.
            Default: ``None``.

    Returns:
        :obj:`DataLoader`: The constructed data loader.
    """
    if isinstance(cfg, DataLoader):
        return cfg

    _cfg = cfg.copy()

    if isinstance(_cfg, dict):
        loader_cfg = _cfg.pop('loader', dict())
    else:
        loader_cfg = dict()

    dataset = build_dataset(_cfg, **kwargs)

    rank, world_size = get_dist_info(group=group)
    num_workers = loader_cfg.get('num_workers', 0)

    if is_distributed() if dist is None else dist:
        loader_cfg['sampler'] = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=loader_cfg.pop('shuffle', False),
            seed=seed,
            drop_last=loader_cfg.pop('drop_last', False))

    data_loader = DataLoader(
        dataset,
        collate_fn=collate,
        worker_init_fn=None if seed is None else partial(
            _init_fn, num_workers=num_workers, rank=rank, seed=seed),
        **loader_cfg)

    return data_loader


def build_hook(cfg, **kwargs):
    """
    Build a hook from a dict or str. The hook should be registered in
    :obj:`HOOKS`.

    Args:
        cfg (dict | str): The config or name of the hook.

    Returns:
        :obj:`Hook`: The constructed hook.
    """
    return HOOKS.build(cfg, **kwargs)
