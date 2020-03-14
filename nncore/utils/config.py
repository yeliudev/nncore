# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp
import sys
from collections import OrderedDict
from collections.abc import MutableSequence
from copy import deepcopy
from importlib import import_module

import nncore
from .misc import bind_getter


class CfgNode(OrderedDict):
    """
    An extended `dict` class with several practical methods.

    This class is an extension of the built-in type `OrderedDict`. The
    interface is the same as a dict object and also allows access config values
    as attributes. The input to the init method could be either a single dict
    or several named parameters.
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('too many arguments')

        if len(args) == 1:
            if isinstance(args[0], (Config, dict)):
                kwargs.update(args[0])
            else:
                raise TypeError("unsupported type '{}'".format(type(args[0])))

        for k, v in kwargs.items():
            self[k] = self._parse_value(v)

    def _parse_value(self, value):
        if isinstance(value, dict):
            value = self.__class__(**value)
        elif isinstance(value, (list, tuple)):
            value = type(value)(self._parse_value(v) for v in value)
        elif isinstance(value, MutableSequence):
            value = [self._parse_value(v) for v in value]
        return value

    def __setitem__(self, key, value):
        super(CfgNode, self).__setitem__(key, self._parse_value(value))

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        if hasattr(self.__class__, key):
            raise AttributeError("attribute '{}' is read-only".format(key))
        self[key] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __repr__(self):
        return super(OrderedDict, self).__repr__()

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[deepcopy(key, memo)] = deepcopy(value, memo)
        return other

    def copy(self):
        return deepcopy(self)

    def update(self, *args, **kwargs):
        other = {}
        if len(args) > 1:
            raise TypeError('too many arguments')
        elif len(args) == 1:
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or (not isinstance(self[k], dict))
                    or (not isinstance(v, dict))):
                self[k] = self._parse_value(v)
            else:
                self[k].update(v)

    def setdefault(self, key, value):
        if key in self:
            return self[key]
        else:
            self[key] = value
            return value

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else item
                    for item in value)
            else:
                base[key] = value
        return base

    def to_json(self):
        return nncore.dumps(self.to_dict(), file_format='json', indent=2)


@bind_getter('filename', 'text')
class Config(object):
    """
    A facility for better :class:`dict` objects.

    This class is a wrapper for :class:`CfgNode` which can be initialized from
    a config file. Users can use the static method :meth:`Config.from_file` to
    create a `Config` object.
    """

    @staticmethod
    def from_file(filename):
        """
        Initialize a Config object from a file.

        Args:
            filename (str): key of the config file. Currently supported
                formats include `py`, `json` and `yaml/yml`.

        Returns:
            cfg (Config): the created `Config` object
        """
        filename = osp.abspath(osp.expanduser(filename))
        nncore.file_exist(filename, raise_error=True)

        file_format = filename.split('.')[-1]
        if file_format == 'py':
            module_name = osp.basename(filename)[:-3]
            config_dir = osp.dirname(filename)
            if '.' in module_name:
                raise ValueError('dots are not allowed in the file path')
            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg = {
                k: v
                for k, v in mod.__dict__.items()
                if not k.startswith('__') or not k.endswith('__')
            }
        elif file_format in ['json', 'yml', 'yaml']:
            cfg = nncore.load(filename)
        else:
            raise TypeError('unsupported format: {}'.format(file_format))

        return Config(cfg=cfg, filename=filename)

    def __init__(self, cfg=None, filename=None):
        if isinstance(cfg, (type(self), dict)):
            _cfg = CfgNode(cfg)
        elif cfg is None:
            _cfg = CfgNode()
        else:
            raise TypeError("unsupported type '{}'".format(type(cfg)))

        super(Config, self).__setattr__('_cfg', _cfg)
        super(Config, self).__setattr__('_filename', filename)

        if filename is not None:
            with open(filename, 'r') as f:
                super(Config, self).__setattr__('_text', f.read())
        else:
            super(Config, self).__setattr__('_text', '')

    def __repr__(self):
        return 'Config(filename: {}): {}'.format(self._filename,
                                                 self._cfg.__repr__())

    def __len__(self):
        return len(self._cfg)

    def __getattr__(self, key):
        return getattr(self._cfg, key)

    def __getitem__(self, key):
        return self._cfg.__getitem__(key)

    def __setattr__(self, key, value):
        self._cfg.__setattr__(key, value)

    def __setitem__(self, key, value):
        self._cfg.__setitem__(key, value)

    def __iter__(self):
        return iter(self._cfg)

    def setdefault(self, key, value):
        return self._cfg.setdefault(key, value)

    def to_dict(self):
        return self._cfg.to_dict()
