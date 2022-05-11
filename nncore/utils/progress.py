# Copyright (c) Ye Liu. Licensed under the MIT License.

from math import ceil
from shutil import get_terminal_size

from .timer import Timer


class ProgressBar(object):
    """
    A progress bar which can show the state of a progress. It only takes
    effect in the main process.
    """

    _wb = '\r[{{}}] {}/{}, {:.1f} task/s, elapsed: {}, eta: {}{}'
    _ob = '\rcompleted: {}, elapsed: {}, {:.1f} tasks/s'

    def __init__(self, num_tasks=None, active=None):
        self._task_num = num_tasks
        self._completed = 0

        if active is None:
            try:
                from nncore.engine import is_main_process
                self._active = is_main_process()
            except ImportError:
                self._active = True
        else:
            self._active = active

        if self._active:
            if self._task_num is not None:
                msg = self._wb.format(0, self._task_num, 0, 0, 0, '')
                msg = msg.format(' ' * self._get_bar_width(msg))
            else:
                msg = self._ob.format(0, 0, 0)

            print(msg, end='')
            self._last_length = len(msg)

            self._timer = Timer()

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
            ela_str = self._get_time_str(ceil(ela))

            if self._task_num is not None:
                perc = self._completed / float(self._task_num)
                eta = int(ela * (1 - perc) / perc + 0.5)
                eta_str = self._get_time_str(ceil(eta))
                msg = self._wb.format(
                    self._completed, self._task_num, fps, ela_str, eta_str,
                    '\n' if self._task_num == self._completed else '')
                bar_width = self._get_bar_width(msg)
                mark_width = int(bar_width * perc)
                chars = '>' * mark_width + ' ' * (bar_width - mark_width)
                msg = msg.format(chars)
            else:
                msg = self._ob.format(self._completed, ela_str, fps)

            print(msg.ljust(self._last_length), end='')
            self._last_length = len(msg)
