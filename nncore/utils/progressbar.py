# Copyright (c) Ye Liu. All rights reserved.

from functools import partial
from math import ceil
from shutil import get_terminal_size

from .timer import Timer


class ProgressBar(object):
    """
    A progress bar which can show the state of a progress.
    """

    _with_bar = '\r[{{}}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s{}'
    _wo_bar = '\rcompleted: {}, elapsed: {}s, {:.1f} tasks/s'

    def __init__(self, task_num=None, distributed=False):
        self.write = partial(print, end='')
        self._task_num = task_num
        self._completed = 0
        self._timer = Timer()

        if distributed:
            from nncore.ops.comm import is_main_process
            self.enabled = is_main_process()
        else:
            self.enabled = True

        if self.enabled:
            if self._task_num is not None:
                msg = self._with_bar.format(0, self._task_num, 0, 0, 0, '')
                bar_width = self._get_bar_width(msg)
                self.write(msg.format(' ' * bar_width))
            else:
                self.write(self._wo_bar.format(0, 0, 0))

    def _get_bar_width(self, msg):
        width, _ = get_terminal_size()
        bar_width = min(int(width - len(msg)) + 2, int(width * 0.6), 50)
        return max(2, bar_width)

    def update(self):
        if not self.enabled:
            return

        self._completed += 1
        elapsed = self._timer.seconds()
        fps = self._completed / elapsed

        if self._task_num is not None:
            percentage = self._completed / float(self._task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = self._with_bar.format(
                self._completed, self._task_num, fps, ceil(elapsed), eta,
                '\n' if self._task_num == self._completed else '')

            bar_width = self._get_bar_width(msg)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.write(msg.format(bar_chars))
        else:
            self.write(
                self._wo_bar.format(self._completed, ceil(elapsed), fps))
