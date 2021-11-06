# Copyright (c) Ye Liu. All rights reserved.

from torch.utils.data.dataset import Dataset

import nncore
from .builder import DATASETS, build_dataset


@DATASETS.register()
@nncore.bind_getter('times')
@nncore.bind_method('_dataset', ['set_state', 'evaluate'])
class RepeatDataset(Dataset):
    """
    A wrapper of repeated dataset.

    The length of repeated dataset will be ``times`` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using this class can reduce the data loading time among epochs.

    Args:
        dataset (:obj:`Dataset` | cfg | str): The dataset or config of dataset
            to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        if not isinstance(dataset, Dataset):
            dataset = build_dataset(dataset)

        self._dataset = dataset
        self._times = times

        self.CLASSES = getattr(dataset, 'CLASSES', None)

    def __getitem__(self, idx):
        return self.dataset[idx % len(self._dataset)]

    def __len__(self):
        state = getattr(self._dataset, 'state', None)
        times = 1 if state in ('val', 'test') else self.times
        return len(self._dataset) * times

    @property
    def dataset(self):
        return self._dataset
