# Copyright (c) Ye Liu. All rights reserved.

from torch.utils.data import DataLoader, DistributedSampler

from nncore import Registry
from nncore.dataset import build_dataset
from nncore.parallel import collate
from .comm import get_rank, get_world_size, is_distributed
from .utils import set_random_seed

HOOKS = Registry('hook')


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

    rank = get_rank(group=group)
    num_workers = loader_cfg.get('num_workers', 0)

    def _init_fn(worker_id):
        set_random_seed(seed=seed + worker_id + rank * num_workers)

    dataset = build_dataset(_cfg, **kwargs)

    if is_distributed() if dist is None else dist:
        shuffle = loader_cfg.pop('shuffle', False)
        drop_last = loader_cfg.pop('drop_last', False)

        world_size = get_world_size(group=group)
        loader_cfg['sampler'] = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last)

    data_loader = DataLoader(
        dataset,
        collate_fn=collate,
        worker_init_fn=None if seed is None else _init_fn,
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
