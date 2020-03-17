# Copyright (c) Ye Liu. All rights reserved.

from time import perf_counter


class Timer(object):
    """
    A flexible timer class.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the timer.
        """
        self._start = perf_counter()
        self._paused = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self):
        """
        Pause the timer.
        """
        if self._paused is not None:
            raise ValueError('Trying to pause a Timer that is already paused!')

        self._paused = perf_counter()

    def is_paused(self):
        """
        Returns:
            paused (bool): whether the timer is currently paused
        """
        return self._paused is not None

    def resume(self):
        """
        Resume the timer.
        """
        if self._paused is None:
            raise ValueError('Trying to resume a Timer that is not paused!')

        self._total_paused += perf_counter() - self._paused
        self._paused = None
        self._count_start += 1

    def seconds(self, reset=False):
        """
        Returns:
            seconds (float): the total number of seconds since the start/reset
                of the timer, excluding the time when the timer is paused
        """
        if self._paused is not None:
            end_time = self._paused
        else:
            end_time = perf_counter()

        return end_time - self._start - self._total_paused

    def avg_seconds(self):
        """
        Returns:
            seconds (float): the average number of seconds between every
                start/reset and pause
        """
        return self.seconds() / self._count_start
