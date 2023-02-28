# Copyright (c) Ye Liu. Licensed under the MIT License.

from torch.utils.data import Dataset as _Dataset


class Dataset(_Dataset):
    """
    A :obj:`torch.utils.data.Dataset` class that supports state configurations.
    The `set_state` method is expected to be called by :obj:`Engine` to trigger
    state transformations.
    """

    def set_state(self, state):
        self.state = state
