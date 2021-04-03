# Copyright (c) Ye Liu. All rights reserved.

from copy import deepcopy


def bind_getter(*vars):
    """
    A syntactic sugar for automatically binding getters for classes. This
    method is expected to be used as a decorator.

    Args:
        *vars: strings indicating the member variables to be binded with
            getters. The name of member variables are expected to start with an
            underline (e.g. `_name` or `_epoch`).

    Example:
        >>> @bind_getter('name', 'depth')
        >>> class Backbone:
        ...
        ...     _name = 'ResNet'
        ...     _depth = 50
        ...
        >>> # Equals to:
        >>> class Backbone:
        ...
        ...     _name = 'ResNet'
        ...     _depth = 50
        ...
        ...     @property
        ...     def name(self):
        ...         return self._name
        ...
        ...     @property
        ...     def depth(self):
        ...         return self._depth
    """

    def _wrapper(cls):
        for var in vars:
            meth = property(lambda self, key='_{}'.format(var): deepcopy(
                getattr(self, key, None)))
            setattr(cls, var, meth)
        return cls

    return _wrapper


def bind_method(key, methods):
    """
    A syntactic sugar for automatically binding methods of classes with their
    attributes. This method is expected to be used as a decorator.

    Args:
        key (str): The key of the attribute to be binded.
        methods (list[str]): The list of method names to be binded.

    Example:
        >>> @bind_method('_cfg', ['get', 'pop'])
        >>> class Config:
        ...
        ...     _name = 'model config'
        ...     _cfg = dict()
        ...
        >>> # Equals to:
        >>> class Config:
        ...
        ...     _name = 'model config'
        ...     _cfg = dict()
        ...
        ...     def get(self, *args):
        ...         return self._cfg.get(*args)
        ...
        ...     def pop(self, *args):
        ...         return self._cfg.pop(*args)
    """

    def _decorator(cls):
        for meth in methods:
            if meth in ('__getattr__', '__setattr__'):
                raise KeyError(
                    "method '{}' can not be binded automatically".format(meth))

            def _wrapper(self, *args, _meth=meth, **kwargs):
                return getattr(getattr(self, key), _meth)(*args, **kwargs)

            setattr(cls, meth, _wrapper)

        return cls

    return _decorator
