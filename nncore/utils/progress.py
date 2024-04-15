# Copyright (c) Ye Liu. Licensed under the MIT License.

import math
from shutil import get_terminal_size

from .timer import Timer


class ProgressBar(object):
    """
    A progress bar which showing the state of a progress.

    Args:
        iterable (iterable | None, optional): The iterable object to decorate
            with a progress bar. Default: ``None``.
        num_tasks (int | None, optional): The number of expected iterations.
            If not specified, the length of iterable object will be used.
            Default: ``None``.
        percentage (bool, optional): Whether to display the percentage instead
            of raw task numbers. Default: ``False``.
        active (bool | None, optional): Whether the progress bar is active. If
            not specified, only the main process will show the progree bar.
            Default: ``None``.

    Example:
        >>> for item in ProgressBar(range(10)):
        ...     # do processing

        >>> prog_bar = ProgressBar(num_tasks=10)
        >>> for item in range(10):
        ...     # do processing
        ...     prog_bar.update()
    """

    _pb = '\r[{{}}] {}, elapsed: {}, eta: {}{}'
    _wb = '\r[{{}}] {}/{}, {:.1f} task/s, elapsed: {}, eta: {}{}'
    _ob = '\rCompleted: {}, elapsed: {}, {:.1f} tasks/s'

    def __init__(self,
                 iterable=None,
                 num_tasks=None,
                 percentage=False,
                 active=None):
        self._iterable = iterable
        self._num_tasks = num_tasks or (len(iterable) if hasattr(
            iterable, '__len__') else None)
        self._percentage = percentage
        self._completed = 0

        if self._percentage:
            assert self._num_tasks is not None

        if active is None:
            try:
                from nncore.engine import is_main_process
                self._active = is_main_process()
            except ImportError:
                self._active = True
        else:
            self._active = active

        if self._active:
            if self._percentage:
                msg = self._pb.format('0%', 0, 0, '')
                msg = msg.format(' ' * self._get_bar_width(msg))
            elif self._num_tasks is not None:
                msg = self._wb.format(0, self._num_tasks, 0, 0, 0, '')
                msg = msg.format(' ' * self._get_bar_width(msg))
            else:
                msg = self._ob.format(0, 0, 0)

            print(msg, end='')
            self._last_length = len(msg)

            self._timer = Timer()

    def __iter__(self):
        for item in self._iterable:
            yield item
            self.update()

    def _get_bar_width(self, msg):
        width, _ = get_terminal_size()
        bar_width = min(int(width - len(msg)) + 2, int(width * 0.6), 40)
        return max(2, bar_width)

    def _get_time_str(self, second):
        if second >= 86400:
            day = second // 86400
            second -= (day * 86400)
            day = '{}d'.format(day)
        else:
            day = ''

        if second >= 3600:
            hour = second // 3600
            second -= (hour * 3600)
            hour = '{}h'.format(hour)
        else:
            hour = ''

        if second >= 60:
            minute = second // 60
            second -= minute * 60
            minute = '{}m'.format(minute)
        else:
            minute = ''

        if second > 0:
            second = '{}s'.format(second)
        else:
            second = ''

        time_str = '{}{}{}{}'.format(day, hour, minute, second) or '0s'
        return time_str

    def update(self, times=1):
        if not self._active:
            return

        for _ in range(times):
            self._completed += 1
            ela = self._timer.seconds()
            fps = self._completed / ela
            ela_str = self._get_time_str(math.ceil(ela))

            if self._num_tasks is not None:
                perc = self._completed / self._num_tasks
                eta = int(ela * (1 - perc) / perc + 0.5)
                eta_str = self._get_time_str(math.ceil(eta))
                end_str = '\n' if self._num_tasks == self._completed else ''
                if self._percentage:
                    msg = self._pb.format('{}%'.format(round(perc * 100, 1)),
                                          ela_str, eta_str, end_str)
                else:
                    msg = self._wb.format(self._completed, self._num_tasks,
                                          fps, ela_str, eta_str, end_str)
                bar_width = self._get_bar_width(msg)
                mark_width = int(bar_width * perc)
                chars = '>' * mark_width + ' ' * (bar_width - mark_width)
                msg = msg.format(chars)
            else:
                msg = self._ob.format(self._completed, ela_str, fps)

            print(msg.ljust(self._last_length), end='')
            self._last_length = len(msg)
