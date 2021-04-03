# Copyright (c) Ye Liu. All rights reserved.

from collections import defaultdict

from .binder import bind_getter, bind_method


@bind_getter('name')
@bind_method('_items', ['__contains__', '__len__', 'get', 'pop', 'items'])
class Registry(object):
    """
    A registry to map strings to objects.

    Records in the :obj:`self.items` maintain the registry of objects. For each
    record, the key is the object name and the value is the object itself. The
    method :obj:`self.register` can be used as a decorator or a normal
    function.

    Args:
        name (str): Name of the registry.

    Example:
        >>> backbones = Registry('backbone')
        >>> @backbones.register()
        >>> class ResNet(object):
        ...     pass

        >>> backbones = Registry('backbone')
        >>> class ResNet(object):
        ...     pass
        >>> backbones.register(ResNet)
    """

    def __init__(self, name):
        self._name = name
        self._items = dict()
        self._groups = defaultdict(list)

    def __getattr__(self, key):
        if key in self._items:
            return self._items[key]
        else:
            raise AttributeError(
                "registry object has no attribute '{}'".format(key))

    def __repr__(self):
        return "{}(name='{}', items={})".format(self.__class__.__name__,
                                                self._name,
                                                list(self._items.keys()))

    def _register(self, obj, name=None, group=None):
        if name is None:
            name = obj.__name__

        if name in self._items:
            raise KeyError('{} is already registered in {}'.format(
                name, self._name))

        self._items[name] = obj

        if group is not None:
            self.set_group(name, group)

    def register(self, obj=None, name=None, group=None):
        if obj is not None:
            if isinstance(name, (list, tuple)):
                for n in name:
                    self._register(obj, name=n, group=group)
            else:
                self._register(obj, name=name, group=group)
            return

        def _wrapper(obj):
            self._register(obj, name=name, group=group)
            return obj

        return _wrapper

    def set_group(self, name, group):
        if name not in self._items:
            raise KeyError('{} is not registered in {}'.format(
                name, self._name))

        if isinstance(group, (list, tuple)):
            for g in group:
                self.set_group(name, g)
            return

        self._groups[group].append(name)

    def groups(self):
        return self._groups.keys()

    def group(self, name, default=None):
        return self._groups.get(name, default)


def build_object(cfg, parent, default=None, **kwargs):
    """
    Build an object from a dict.

    The dict must contain a key ``type``, which is a indicating the object
    type. Remaining fields are treated as the arguments for constructing the
    object.

    Args:
        cfg (any): The object, object config or object name.
        parent (any): The module or a list of modules which may contain the
            expected object.
        default (any, optional): The default value when the object is not
            found. Default: ``None``.

    Returns:
        any: The constructed object.
    """
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    elif not isinstance(cfg, dict):
        return cfg

    if isinstance(parent, (list, tuple)):
        for p in parent:
            obj = build_object(cfg, p, **kwargs)
            if obj is not None:
                return obj
        return default

    _cfg = cfg.copy()
    _cfg.update(kwargs)
    obj_type = _cfg.pop('type')

    if hasattr(parent, 'get'):
        obj_cls = parent.get(obj_type)
    else:
        obj_cls = getattr(parent, obj_type, None)

    return obj_cls(**_cfg) if obj_cls is not None else default
