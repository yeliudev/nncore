# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import torch

import nncore


@nncore.bind_getter('max_size')
@nncore.bind_method('_data', ['get', 'pop', 'keys', 'values', 'items'])
class Buffer(object):
    """
    A buffer that tracks a series of values and provide access to smoothed
    scalar values over a window.

    Args:
        max_size (int, optional): Maximal number of internal values that can
            be stored in the buffer. When the capacity of the buffer is
            exhausted, old values will be removed. Default: ``100000``.
        logger (:obj:`logging.Logger` or str or None, optional): The potential
            logger or name of the logger to use. Default: ``None``.
    """

    def __init__(self, max_size=100000, logger=None):
        self._max_size = max_size
        self._logger = logger
        self._data = OrderedDict()

    def update(self, key, value, warning=False):
        """
        Add a new value. If the length of the buffer exceeds
        :obj:`self._max_size`, the oldest element will be removed from the
        buffer.

        Args:
            key (str): The key of the values.
            value (number): The new value of the values.
            warning (bool, optional): Whether to display warning when removing
                values. Default: ``False``.
        """
        if key not in self._data:
            self._data[key] = []
        elif not key.startswith('_') and len(
                self._data[key]) == self._max_size:
            if warning:
                nncore.log_or_print(
                    "Number of '{}' values in the buffer exceeds max size "
                    "({}), removing the oldest element".format(
                        key, self._max_size),
                    self._logger,
                    log_level='WARNING')
            self._data[key].pop(0)

        self._data[key].append(value)

    def count(self, key):
        """
        Return the number of values according to the key.

        Args:
            key (str): The key of the values.
        """
        return len(self._data[key])

    def clear(self):
        """
        Remove all values from the buffer.
        """
        self._data = OrderedDict()

    def latest(self, key):
        """
        Return the latest value in the buffer.

        Args:
            key (str): The key of the values.
        """
        return self._data[key][-1]

    def median(self, key, window_size=None):
        """
        Return the median of the latest ``window_size`` values in the buffer.

        Args:
            key (str): The key of the values.
            window_size (int or None, optional): The window size of the values
                to be computed. If not specified, all the values will be taken
                into account. Default: ``None``.

        Returns:
            float: The median of the latest ``window_size`` values.
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
        Return the mean of the latest ``window_size`` values in the buffer.

        Args:
            key (str): The key of the values.
            window_size (int or None, optional): The window size of the values
                to be computed. If not specified, all the values will be taken
                into account. Default: ``None``.

        Returns:
            float: The mean of the latest ``window_size`` values.
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
        Return the sum of the latest ``window_size`` values in the buffer.

        Args:
            key (str): The key of the values.
            window_size (int or None, optional): The window size of the values
                to be computed. If not specified, all the values will be taken
                into account. Default: ``None``.

        Returns:
            float: The sum of the latest ``window_size`` values.
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
        Return the average of the latest ``window_size`` values in the buffer.
        Note that since not all the values in the buffer are count from the
        same number of samples, the exact average of these values should be
        computed with the number of samples.

        Args:
            key (str): The key of the values.
            by (str, optional): The key of number of samples. Default:
                ``'_num_samples'``.
            window_size (int or None, optional): The window size of the values
                to be computed. If not specified, all the values will be taken
                into account. Default: ``None``.

        Returns:
            float: The average of the latest ``window_size`` values.
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
