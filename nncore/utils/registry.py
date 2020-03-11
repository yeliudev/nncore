# Copyright (c) Ye Liu. All rights reserved.

from inspect import isclass


class Registry(object):
    """
    A registry to map strings to classes.

    Records in the `self._modules` maintain the registry of classes. For each
    record, the key is the class name and the value is the class itself. The
    method `self.register_module` can be used as a decorator or a normal
    function.

    Example:
        >>> backbones = Registry('backbone')
        >>> @backbones.register_module
        >>> class ResNet(object):
        >>>     pass

    Example:
        >>> backbones = Registry('backbone')
        >>> class ResNet(object):
        >>>     pass
        >>> backbones.register_module(ResNet)

    Args:
        name (str): registry name
    """

    def __init__(self, name):
        self._name = name
        self._modules = dict()

    def __len__(self):
        return len(self._modules)

    def __getattr__(self, key):
        return self._modules.get(key, None)

    def __repr__(self):
        return '{}(name={}, modules={})'.format(self.__class__.__name__,
                                                self._name,
                                                list(self._modules.keys()))

    @property
    def name(self):
        return self._name

    @property
    def modules(self):
        return self._modules

    def register_module(self, module_cls):
        if not isclass(module_cls):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_cls)))
        module_name = module_cls.__name__
        if module_name in self._modules:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._modules[module_name] = module_cls
        return module_cls

    def get(self, key):
        return self.__getattr__(key)


def build_object(cfg, parent=None, default_args=None):
    """
    Initialize an object from a dict.

    The dict must contain a key 'type', which is a indicating the object type.
    Remaining fields are treated as the arguments for constructing the object.

    Args:
        cfg (dict): object types and arguments
        parent (:class:`module`, optional): the module which may containing
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
