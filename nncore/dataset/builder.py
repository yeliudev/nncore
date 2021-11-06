# Copyright (c) Ye Liu. All rights reserved.

import torch.utils.data.dataset as dataset

from nncore import Registry, build_object

DATASETS = Registry('dataset')


def build_dataset(cfg, *args, **kwargs):
    """
    Build a dataset from a dict. This method searches for datasets in
    :obj:`DATASETS` first, and then fall back to
    :obj:`torch.utils.data.dataset`.

    Args:
        cfg (dict): The config of the dataset.

    Returns:
        :obj:`nn.Module`: The constructed dataset.
    """
    return build_object(cfg, [DATASETS, dataset], args=args, **kwargs)
