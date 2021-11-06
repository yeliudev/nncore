# Copyright (c) Ye Liu. All rights reserved.

from torch.utils.data import DataLoader

from nncore import Registry
from nncore.dataset import build_dataset
from nncore.parallel import collate
from .utils import set_random_seed

HOOKS = Registry('hook')


def build_dataloader(cfg, loader_cfg=None, key=None, seed=None, **kwargs):
    """
    Build a data loader from a dict. The dataset should be registered in
    :obj:`DATASETS`.

    Args:
        cfg (dict): The config of the dataset.
        loader_cfg (dict | None, optional): The config of the data loader.
            Default: ``None``.
        key (str | None, optional): The key for of the data loader config.
            Default: ``None``.
        seed (int | None, optional): The random seed to use. Default: ``None``.

    Returns:
        :obj:`DataLoader`: The constructed data loader.
    """
    if isinstance(cfg, DataLoader):
        return cfg

    def _init_fn(worker_id):
        set_random_seed(seed=seed + worker_id)

    if loader_cfg is not None:
        loader_cfg = loader_cfg.get(key, dict())
    elif isinstance(cfg, dict):
        loader_cfg = cfg.pop('loader', dict())
    else:
        loader_cfg = dict()

    dataset = build_dataset(cfg, **kwargs)
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
