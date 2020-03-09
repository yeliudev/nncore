# Copyright (c) Ye Liu. All rights reserved.

from time import time


class Timer(object):
    """
    A flexible Timer class.

    :Example:

    >>> import time
    >>> import mmcv
    >>> with mmcv.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = mmcv.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        if not self._is_running:
            raise RuntimeError('Timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        if not self._is_running:
            raise RuntimeError('Timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur
