# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import torch

import nncore


@nncore.bind_getter('max_size')
class Buffer(object):
    """
    A buffer that can track a series of values and provide access to smoothed
    scalar values over a window.
    """

    def __init__(self, max_size=100000, logger=None):
        """
        Args:
            max_length (int, optional): maximal number of values that can be
                stored in the buffer. When the capacity of the buffer is
                exhausted, old values will be removed.
            logger (:obj:`logging.Logger` or str or None, optional): the
                potential logger or name of the logger to be used
        """
        self._max_size = max_size
        self._logger = logger
        self._data = OrderedDict()

    def __iter__(self):
        return self._data.__iter__()

    def keys(self):
        """
        Return the keys in the buffer.
        """
        return self._data.keys()

    def values(self):
        """
        Return the values in the buffer.
        """
        return self._data.values()

    def items(self):
        """
        Return the keys and values in the buffer.
        """
        return self._data.items()

    def get(self, key, default=None):
        """
        Return the values in the buffer according to the key.

        Args:
            key (str): the key of the values

        Returns:
            values (list): the values in the buffer according to the key
        """
        return self._data.get(key, default=default)

    def update(self, key, value, warning=True):
        """
        Add a new value. If the length of the buffer exceeds `self._max_size`,
        the oldest element will be removed from the buffer.

        Args:
            key (str): the key of the values
            value (number): the value of the values
            warning (bool, optional): whether to show warning when deleting
                values
        """
        if key not in self._data:
            self._data[key] = []
        elif len(self._data[key]) == self._max_size:
            if warning:
                nncore.log_or_print(
                    "Number of '{}' values in the buffer exceeds max size "
                    "({}), removing the oldest element".format(
                        key, self._max_size),
                    self._logger,
                    log_level='WARNING')
            self._data[key].pop(0)

        self._data[key].append(value)

    def pop(self, key, default=None):
        """
        Return and remove the values in the buffer according to the key.

        Args:
            key (str): the key of the values

        Returns:
            values (list): the values in the buffer according to the key
        """
        return self._data.pop(key, default=default)

    def count(self, key):
        """
        Return the number of values according to the key.

        Args:
            key (str): the key of the values
        """
        return len(self._data[key])

    def clear(self):
        """
        Remove all values from the buffer.
        """
        self._data = OrderedDict()

    def latest(self, key):
        """
        Return the latest value added to the buffer.

        Args:
            key (str): the key of the values
        """
        return self._data[key][-1]

    def median(self, key, window_size=None):
        """
        Return the median of the latest `window_size` values in the buffer.

        Args:
            key (str): the key of the values
            window_size (int or None, optional): the window_size of the values
                to be computed

        Returns:
            median (number): the median of the latest `window_size` values
        """
        if window_size is None or window_size > len(self._data[key]):
            window_size = len(self._data[key])

        if isinstance(self._data[key][0], dict):
            data = nncore.to_dict_of_list(self._data[key][-window_size:])
            median = {
                k: torch.Tensor(v).median().item()
                for k, v in data.items()
            }
        else:
            median = torch.Tensor(
                self._data[key][-window_size:]).median().item()

        return median

    def mean(self, key, window_size=None):
        """
        Return the mean of the latest `window_size` values in the buffer.

        Args:
            key (str): the key of the values
            window_size (int or None, optional): the window_size of the values
                to be computed

        Returns:
            mean (number): the mean of the latest `window_size` values
        """
        if window_size is None or window_size > len(self._data[key]):
            window_size = len(self._data[key])

        if isinstance(self._data[key][0], dict):
            data = nncore.to_dict_of_list(self._data[key][-window_size:])
            mean = {k: torch.Tensor(v).mean().item() for k, v in data.items()}
        else:
            mean = torch.Tensor(self._data[key][-window_size:]).mean().item()

        return mean

    def sum(self, key, window_size=None):
        """
        Return the sum of the latest `window_size` values in the buffer.

        Args:
            key (str): the key of the values
            window_size (int or None, optional): the window_size of the values
                to be computed

        Returns:
            sum (number): the sum of the latest `window_size` values
        """
        if window_size is None or window_size > len(self._data[key]):
            window_size = len(self._data[key])

        if isinstance(self._data[key][0], dict):
            data = nncore.to_dict_of_list(self._data[key][-window_size:])
            sum = {k: torch.Tensor(v).sum().item() for k, v in data.items()}
        else:
            sum = torch.Tensor(self._data[key][-window_size:]).sum().item()

        return sum

    def avg(self, key, by='_num_samples', window_size=None):
        """
        Return the average of the latest `window_size` values in the buffer.
        Note that since not all the values in the buffer are count from the
        same number of samples, the exact average of these values should be
        computed with the number of samples.

        Args:
            key (str): the key of the values
            by (str, optional): the key of the number of samples
            window_size (int or None, optional): the window_size of the values
                to be computed

        Returns:
            avg (number): the average of the latest `window_size` values
        """
        if window_size is None or window_size > len(self._data[key]):
            window_size = len(self._data[key])

        num_samples = torch.Tensor(self._data[by][-window_size:])

        if isinstance(self._data[key][0], dict):
            data = nncore.to_dict_of_list(self._data[key][-window_size:])
            avg = {
                k: ((torch.Tensor(v) * num_samples).sum() /
                    num_samples.sum()).item()
                for k, v in data.items()
            }
        else:
            scalar = torch.Tensor(self._data[key][-window_size:])
            avg = ((scalar * num_samples).sum() / num_samples.sum()).item()

        return avg
