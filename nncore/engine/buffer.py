# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import torch


class Buffer(object):
    """
    A buffer that can track a series of values and provide access to smoothed
    scalar values over a window.
    """

    def __init__(self, max_size=100000):
        """
        Args:
            max_length (int, optional): maximal number of values that can be
                stored in the buffer. When the capacity of the buffer is
                exhausted, old values will be removed.
        """
        self._max_size = max_size
        self._data = OrderedDict()

    def keys(self):
        """
        Return the keys in the buffer.
        """
        return self._data.keys()

    def values(self, key):
        """
        Return the values in the buffer according to the key.

        Args:
            key (str): the key of the values

        Returns:
            values (list): the values in the buffer according to the key
        """
        return self._data[key]

    def update(self, key, value):
        """
        Add a new value. If the length of the buffer exceeds self._max_size,
        the oldest element will be removed from the buffer.

        Args:
            key (str): the key of the values
            value (number): the value of the values
        """
        if key not in self._data:
            self._data[key] = []
        elif len(self._data[key]) == self._max_size:
            self._data[key].pop(0)

        self._data[key].append(value)

    def count(self, key):
        """
        Return the number of values according to the key.

        Args:
            key (str): the key of the values
        """
        return len(self._data[key])

    def clear(self, key=None):
        """
        Clear the buffer according to the key.

        Args:
            key (str or None, optional): the key of the values. If None, clear
                all the values in the buffer.
        """
        if key is None:
            self._data = OrderedDict()
        elif isinstance(key, str) and key in self._data:
            del self._data[key]

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

        return torch.Tensor(self._data[key][-window_size:]).median().item()

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

        return torch.Tensor(self._data[key][-window_size:]).mean().item()

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

        return torch.Tensor(self._data[key][-window_size:]).sum().item()

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

        scalar = torch.Tensor(self._data[key][-window_size:])
        num_samples = torch.Tensor(self._data[by][-window_size:])
        cumsum = scalar * num_samples

        return (cumsum.sum() / num_samples.sum()).item()
