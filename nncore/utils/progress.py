# Copyright (c) Ye Liu. All rights reserved.

from math import ceil
from shutil import get_terminal_size

from .timer import Timer


class ProgressBar(object):
    """
    A progress bar which can show the state of a progress. It only takes
    effect in the main process.
    """

    _wb = '\r[{{}}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {}s{}'
    _ob = '\rcompleted: {}, elapsed: {}s, {:.1f} tasks/s'

    def __init__(self, num_tasks=None):
        self._task_num = num_tasks
        self._completed = 0

        try:
            from nncore.engine import is_main_process
            self._active = is_main_process()
        except ImportError:
            self._active = True

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

    def update(self):
        if not self._active:
            return

        self._completed += 1
        ela = self._timer.seconds()
        fps = self._completed / ela

        if self._task_num is not None:
            perc = self._completed / float(self._task_num)
            msg = self._wb.format(
                self._completed, self._task_num, fps, ceil(ela),
                int(ela * (1 - perc) / perc + 0.5),
                '\n' if self._task_num == self._completed else '')
            bar_width = self._get_bar_width(msg)
            mark_width = int(bar_width * perc)
            chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            msg = msg.format(chars)
        else:
            msg = self._ob.format(self._completed, ceil(ela), fps)

        print(msg.ljust(self._last_length), end='')
        self._last_length = len(msg)
