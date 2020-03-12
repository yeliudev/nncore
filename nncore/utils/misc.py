# Copyright (c) Ye Liu. All rights reserved.

import functools


def bind_getters(*vars):
    """
    A syntactic sugar for automatically binding getters for classes. This
    method is expected to be used as a decorator.

    Args:
        *vars: strings indicating the member variables to be binded with
            getters. The name of member variables are expected to start with an
            underline (e.g. `_name` or `_epoch`).

    Example:
        >>> import nncore
        >>> @nncore.bind_getters('name', 'depth')
        >>> class Backbone:
        >>>     _name = 'ResNet'
        >>>     _depth = 50
    equals to:
        >>> class Backbone:
        >>>     _name = 'ResNet'
        >>>     _depth = 50
        >>>     @property
        >>>     def name(self):
        >>>         return self._name
        >>>     @property
        >>>     def depth(self):
        >>>         return self._depth
    """

    def decorator(cls):

        @functools.wraps(cls)
        def wrapper(*args, **kwargs):
            for var in vars:
                method = functools.partial(
                    lambda self, key: getattr(self, key),
                    key='_{}'.format(var))
                setattr(cls, var, property(method))
            return cls(*args, **kwargs)

        return wrapper

    return decorator
