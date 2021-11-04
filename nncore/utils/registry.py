# Copyright (c) Ye Liu. All rights reserved.

from collections import defaultdict

from .binder import bind_getter
from .misc import recursive


@bind_getter('name')
class Registry(object):
    """
    A registry to map strings to objects.

    Records in the :obj:`self.items` maintain the registry of objects. For each
    record, the key is the object name and the value is the object itself. The
    method :obj:`self.register` can be used as a decorator or a normal
    function.

    Args:
        name (str): Name of the registry.
        parent (list[str] | str | None, optional): The parent registry of list
            of parent registries. Default: ``None``.
        children (list[str] | str | None, optional): The children registry of
            list of children registries. Default: ``None``.

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

    def __init__(self, name, parent=None, children=None):
        self._name = name
        self._items = dict()
        self._groups = defaultdict(list)
        self._children = []

        if isinstance(parent, (list, tuple)):
            for p in parent:
                p.add_children(self)
        elif isinstance(parent, self.__class__):
            parent.add_children(self)

        if children is not None:
            self.add_children(children)

    def __len__(self):
        return len(self._items) + sum(len(c) for c in self._children)

    def __contains__(self, item):
        if item in self._items:
            return True

        for child in self._children:
            if item in child:
                return True

        return False

    def __getattr__(self, key):
        if key in self._items:
            return self._items[key]

        for child in self._children:
            try:
                return child[key]
            except AttributeError:
                pass

        raise AttributeError("registry has no attribute '{}'".format(key))

    def __repr__(self):
        return "{}(name='{}', items={})".format(self.__class__.__name__,
                                                self._name, self.keys())

    def _register(self, obj, name=None, group=None):
        if name is None:
            name = obj.__name__

        if name in self._items:
            raise KeyError('{} is already registered in {}'.format(
                name, self._name))

        self._items[name] = obj

        if group is not None:
            self.set_group(name, group)

    @recursive()
    def add_children(self, children):
        if isinstance(children, self.__class__):
            children = [children]
        self._children += children

    def keys(self):
        keys = list(self._items.keys())

        for child in self._children:
            keys += child.keys()

        return keys

    def get(self, key, default=None):
        obj = self._items.get(key)
        if obj is not None:
            return obj

        for child in self._children:
            obj = child.get(key)
            if obj is not None:
                return obj

        return default

    def pop(self, key, default=None):
        obj = self._items.pop(key)
        if obj is not None:
            return obj

        for child in self._children:
            obj = child.pop(key)
            if obj is not None:
                return obj

        return default

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

    def build(self, cfg, default=None, args=[], **kwargs):
        return build_object(cfg, self, default=default, args=args, **kwargs)


@recursive()
def build_object(cfg, parent, default=None, args=[], **kwargs):
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
        args (list, optional): The argument list used to build the object.

    Returns:
        any: The constructed object.
    """
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    elif cfg is None:
        return default
    elif not isinstance(cfg, dict):
        return cfg

    if isinstance(parent, (list, tuple)):
        for p in parent:
            obj = build_object(cfg, p, args=args, **kwargs)
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

    return obj_cls(*args, **_cfg) if obj_cls is not None else default
