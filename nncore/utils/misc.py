# Copyright (c) Ye Liu. All rights reserved.

from functools import wraps


def recursive(key=0, type='list'):
    """
    A syntactic sugar to make a function recursive. This method is expected to
    be used as a decorator.

    Args:
        key (int or str, optional): The index or name of the argument to
            consider. Default: ``0``.
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
        ...     if isinstance(num, (list, tuple)):
        ...         return [func(n) for n in num]
        ...     else:
        ...         return num + 1
        ...
        >>> @recursive(key='name', type='dict')
        >>> def func(value, name=['a', 'b']):
        ...     return dict(name=value)
        ...
        >>> # Equals to:
        >>> def func(value, name=['a', 'b']):
        ...     if isinstance(name, (list, tuple)):
        ...         out_dict = dict()
        ...         for n in name:
        ...             out_dict.update(func(value, n))
        ...         return out_dict
        ...     else:
        ...         return dict(name=value)
    """
    assert isinstance(key, (int, str))
    assert type in ('list', 'tuple', 'dict')

    def _decorator(func):

        @wraps(func)
        def _wrapper(*args, **kwargs):
            arg = args[key] if isinstance(key, int) else kwargs[key]

            if isinstance(arg, (list, tuple)):
                out = []
                for a in arg:
                    if isinstance(key, int):
                        args[key] = a
                    else:
                        kwargs[key] = a

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
