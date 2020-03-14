# Copyright (c) Ye Liu. All rights reserved.

from inspect import isclass

from .misc import bind_getter


@bind_getter('name', 'items')
class Registry(object):
    """
    A registry to map strings to classes.

    Records in the `self._items` maintain the registry of classes. For each
    record, the key is the class name and the value is the class itself. The
    method `self.register` can be used as a decorator or a normal function.

    Example:
        >>> backbones = Registry('backbone')
        >>> @backbones.register
        >>> class ResNet(object):
        >>>     pass

        >>> backbones = Registry('backbone')
        >>> class ResNet(object):
        >>>     pass
        >>> backbones.register(ResNet)

    Args:
        name (str): registry name
    """

    def __init__(self, name):
        self._name = name
        self._items = {}

    def __len__(self):
        return len(self._items)

    def __getattr__(self, key):
        return self._items.get(key, None)

    def __repr__(self):
        return '{}(name={}, items={})'.format(self.__class__.__name__,
                                              self._name,
                                              list(self._items.keys()))

    def register(self, obj_cls):
        if not isclass(obj_cls):
            raise TypeError('obj_cls must be a class, but got {}'.format(
                type(obj_cls)))
        name = obj_cls.__name__
        if name in self._items:
            raise KeyError('{} is already registered in {}'.format(
                name, self.name))
        self._items[name] = obj_cls
        return obj_cls

    def get(self, key):
        return self.__getattr__(key)


def build_object(cfg, parent, default_args=None):
    """
    Initialize an object from a dict.

    The dict must contain a key `type`, which is a indicating the object type.
    Remaining fields are treated as the arguments for constructing the object.

    Args:
        cfg (dict): object types and arguments
        parent (:class:`module`): the module which may containing
            expected object class
        default_args (dict, optional): default arguments for initializing the
            object

    Returns:
        obj (any): object built from the dict
    """
    args = cfg.copy()
    obj_type = args.pop('type')

    obj_cls = getattr(parent, obj_type, None)
    if obj_cls is None:
        raise KeyError("parent '{}' has not attribute '{}'".format(
            parent.__class__.__name__, obj_type))

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_cls(**args)
