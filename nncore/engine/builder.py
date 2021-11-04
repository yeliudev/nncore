# Copyright (c) Ye Liu. All rights reserved.

from torch.utils.data import DataLoader

from nncore import Registry
from nncore.parallel import collate
from .utils import set_random_seed

DATASETS = Registry('dataset')
HOOKS = Registry('hook')


def build_dataloader(cfg, seed=None, **kwargs):
    """
    Build a data loader from a dict. The dataset should be registered in
    :obj:`DATASETS`.

    Args:
        cfg (dict): The config of the data loader.

    Returns:
        :obj:`nn.Module`: The constructed module.
    """

    def _init_fn(worker_id):
        set_random_seed(seed=seed + worker_id)

    _cfg = cfg.pop('loader', dict())

    dataset = DATASETS.build(cfg, **kwargs)
    data_loader = DataLoader(
        dataset, collate_fn=collate, worker_init_fn=_init_fn, **_cfg)

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
