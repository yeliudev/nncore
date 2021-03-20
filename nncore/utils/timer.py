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
        self._paused_time = 0

    def is_paused(self):
        """
        Check whether the timer is paused.
        """
        return self._paused is not None

    def pause(self):
        """
        Pause the timer.
        """
        if self.is_paused():
            raise RuntimeError('the timer that is already paused')

        self._paused = perf_counter()

    def resume(self):
        """
        Resume the timer.
        """
        if not self.is_paused():
            raise RuntimeError('the timer is not paused')

        self._paused_time += perf_counter() - self._paused
        self._paused = None

    def seconds(self):
        """
        Return the total number of seconds since the reset of the timer,
        excluding the time when the timer is paused.
        """
        end_time = self._paused or perf_counter()
        return end_time - self._start - self._paused_time

    def minutes(self):
        """
        Return the total number of minutes since the reset of the timer,
        excluding the time when the timer is paused.
        """
        return self.seconds() / 60

    def hours(self):
        """
        Return the total number of hours since the reset of the timer,
        excluding the time when the timer is paused.
        """
        return self.minutes() / 60
