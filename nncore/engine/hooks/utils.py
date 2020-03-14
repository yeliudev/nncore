# Copyright (c) Ye Liu. All rights reserved.

from functools import wraps


def every_n_epochs(n):
    """
    A decorator to let a hook to be executed every n epochs.
    """

    def _decorator(func):

        @wraps(func)
        def _wrapper(self, engine):
            nonlocal n
            if isinstance(n, str):
                n = getattr(self, n)
            if n <= 0 or (engine.epoch + 1) % n == 0:
                return
            func(self, engine)

        return _wrapper

    return _decorator


def every_n_steps(n):
    """
    A decorator to let a hook to be executed every n steps.
    """

    def _decorator(func):

        @wraps(func)
        def _wrapper(self, engine):
            nonlocal n
            if isinstance(n, str):
                n = getattr(self, n)
            if n <= 0 or (engine.iter + 1) % n == 0:
                return
            func(self, engine)

        return _wrapper

    return _decorator
