# Copyright (c) Ye Liu. All rights reserved.

import inspect
from functools import wraps


def recursive(key=None, type='list'):
    """
    A syntactic sugar to make a function recursive. This method is expected to
    be used as a decorator.

    Args:
        key (str or None, optional): The name of the argument to iterate. If
            not specified, the first argument of the function will be used.
            Default: ``None``.
        type (str, optional): The type of returned object. Expected values
            include ``'list'``, ``'tuple'`` and ``'dict'``. Default:
            ``'list'``.

    Example:
        >>> @recursive()
        >>> def func(num):
        ...     return num + 1
        ...
        >>> # Equals to:
        >>> def func(num):
        ...     if isinstance(num, (list, tuple, range)):
        ...         return [func(n) for n in num]
        ...     else:
        ...         return num + 1
        ...
        >>> @recursive(key='name', type='dict')
        >>> def func(value, name):
        ...     return dict(name=value)
        ...
        >>> # Equals to:
        >>> def func(value, name):
        ...     if isinstance(name, (list, tuple, range)):
        ...         out_dict = dict()
        ...         for n in name:
        ...             out_dict.update(func(value, n))
        ...         return out_dict
        ...     else:
        ...         return dict(name=value)
    """
    assert type in ('list', 'tuple', 'dict')

    def _decorator(func):
        nonlocal key
        params = inspect.signature(func).parameters
        if key is None:
            key = next(iter(params))

        @wraps(func)
        def _wrapper(*args, **kwargs):
            if key in kwargs:
                arg = kwargs[key]
            else:
                idx = list(params).index(key)
                arg = args[idx] if idx < len(args) else params[key].default

            if isinstance(arg, (list, tuple, range)):
                args, out = list(args), []
                for a in arg:
                    if key in kwargs or idx >= len(args):
                        kwargs[key] = a
                    else:
                        args[idx] = a

                    func_out = func(*args, **kwargs)
                    out.append(func_out)

                if type == 'dict':
                    out_dict = dict()
                    for o in out:
                        out_dict.update(o)
                    out = out_dict
                elif type == 'tuple':
                    out = tuple(out)
            else:
                out = func(*args, **kwargs)

            return out

        return _wrapper

    return _decorator
