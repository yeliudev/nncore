# Copyright (c) Ye Liu. All rights reserved.

import sys
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size

from .timer import Timer


class ProgressBar(object):
    """A progress bar which can print the progress"""

    def __init__(self, task_num=0, distributed=False, stream=sys.stdout):
        self.task_num = task_num
        self.stream = stream
        self.completed = 0
        self.start()

        if distributed:
            from nncore.ops.comm import is_main_process
            self.dummy_update = not is_main_process()
        else:
            self.dummy_update = False

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.stream.write('[{}] 0/{}, elapsed: 0s, ETA:'.format(
                ' ' * 50, self.task_num))
        else:
            self.stream.write('completed: 0, elapsed: 0s')
        self.stream.flush()
        self.timer = Timer()

    def update(self):
        if self.dummy_update:
            return

        self.completed += 1
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = '\r[{{}}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s' \
                  ''.format(self.completed, self.task_num, fps,
                            int(elapsed + 0.5), eta)

            bar_width = min(50,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.stream.write(msg.format(bar_chars))
        else:
            self.stream.write(
                'completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                    self.completed, int(elapsed + 0.5), fps))
        self.stream.flush()

        if self.task_num == self.completed:
            self.stream.write('\n')


def track_progress(func, tasks, stream=sys.stdout, **kwargs):
    """
    Track the progress of tasks execution with a progress bar.

    Args:
        func (callable): the function to be applied to each task
        tasks (list or tuple[Iterable, int]): a list of tasks or
            (tasks, total num)

    Returns:
        results (list): the task results
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            "'tasks' must be an iterable object or a (iterator, int) tuple")
    prog_bar = ProgressBar(task_num, stream=stream)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    return results


def track_parallel_progress(func, tasks, nproc=None, stream=sys.stdout):
    """
    Track the progress of parallel task execution with a progress bar

    Args:
        func (callable): the function to be applied to each task
        tasks (list or tuple[Iterable, int]): a list of tasks or
            (tasks, total num)
        nproc (int or None, optional): number of processes

    Returns:
        results (list): the task results
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            "'tasks' must be an iterable object or a (iterator, int) tuple")
    pool = Pool(nproc)
    prog_bar = ProgressBar(task_num, stream=stream)
    results = []
    gen = pool.imap(func, tasks)
    for result in gen:
        results.append(result)
        prog_bar.update()
    pool.close()
    pool.join()
    return results


def track_iter_progress(tasks, stream=sys.stdout, **kwargs):
    """
    Track the progress of tasks iteration or enumeration with a progress bar.

    Args:
        tasks (list or tuple[Iterable, int]): a list of tasks or
            (tasks, total num)

    Yields:
        results (list): the task results
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            "'tasks' must be an iterable object or a (iterator, int) tuple")
    prog_bar = ProgressBar(task_num, stream=stream)
    for task in tasks:
        yield task
        prog_bar.update()
