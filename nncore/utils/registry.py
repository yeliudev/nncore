# Copyright (c) Ye Liu. All rights reserved.

from .misc import bind_getter


@bind_getter('name', 'items')
class Registry(object):
    """
    A registry to map strings to objects.

    Records in the `self._items` maintain the registry of objects. For each
    record, the key is the object name and the value is the object itself. The
    method `self.register` can be used as a decorator or a normal function.

    Example:
        >>> backbones = Registry('backbone')
        >>> @backbones.register()
        >>> class ResNet(object):
        ...     pass

        >>> backbones = Registry('backbone')
        >>> class ResNet(object):
        ...     pass
        >>> backbones.register(ResNet)

    Args:
        name (str): registry name
    """

    def __init__(self, name):
        self._name = name
        self._items = dict()

    def __len__(self):
        return len(self._items)

    def __contains__(self, key):
        return key in self._items

    def __getattr__(self, key):
        if key in self._items:
            return self._items[key]
        else:
            raise AttributeError(
                "Registry object has no attribute '{}'".format(key))

    def __repr__(self):
        return "{}(name='{}', items={})".format(self.__class__.__name__,
                                                self._name,
                                                list(self._items.keys()))

    def _register(self, obj, name=None):
        if name is None:
            name = obj.__name__
        if name in self._items:
            raise KeyError('{} is already registered in {}'.format(
                name, self._name))
        self._items[name] = obj
        return obj

    def register(self, obj=None, name=None):
        if isinstance(obj, (list, tuple)):
            for o in obj:
                self._register(o, name=name)
            return

        if obj is not None:
            self._register(obj, name=name)
            return

        def _wrapper(obj):
            self._register(obj, name=name)
            return obj

        return _wrapper

    def get(self, key, default=None):
        return self._items.get(key, default)

    def pop(self, key):
        return self._items.pop(key)


def build_object(cfg, parent, default=None, **kwargs):
    """
    Initialize an object from a dict.

    The dict must contain a key `type`, which is a indicating the object type.
    Remaining fields are treated as the arguments for constructing the object.

    Args:
        cfg (dict or None): object types and arguments
        parent (any): a module or a sequence of modules which may contain the
            expected object class
        default (any, optional): default return value when the object class not
            found

    Returns:
        obj (any): object built from the dict
    """
    if 'default' in kwargs:
        raise KeyError("argument 'default' is reserved by this method")

    if cfg is None:
        return default

    args = cfg.copy()
    args.update(kwargs)
    obj_type = args.pop('type')

    if isinstance(parent, (list, tuple)):
        for p in parent:
            obj = build_object(cfg, p, **kwargs)
            if obj is not None:
                return obj
        return default
    elif hasattr(parent, 'get'):
        obj_cls = parent.get(obj_type)
    else:
        obj_cls = getattr(parent, obj_type, None)

    return obj_cls(**args) if obj_cls is not None else default
