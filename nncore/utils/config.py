# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp
import sys
from collections import OrderedDict
from copy import deepcopy
from importlib import import_module
from shutil import copyfile
from tempfile import TemporaryDirectory

import nncore
from .misc import bind_getter


class CfgNode(OrderedDict):
    """
    An extended `OrderedDict` class with several practical methods.

    This class is an extension of the built-in type `OrderedDict`. The
    interface is the same as a dict object and also allows access config values
    as attributes. The input to the init method could be either a single dict
    or several named parameters.
    """

    @staticmethod
    def _set_freeze_state(obj, state):
        if isinstance(obj, CfgNode):
            super(CfgNode, obj).__setattr__('_frozen', state)
            for v in obj.values():
                CfgNode._set_freeze_state(v, state)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                CfgNode._set_freeze_state(v, state)

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('too many arguments')

        if len(args) == 1:
            if isinstance(args[0], (Config, dict)):
                kwargs.update(args[0])
            else:
                raise TypeError("unsupported type '{}'".format(type(args[0])))

        super(CfgNode, self).__setattr__('_frozen', False)
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key, value):
        self._check_freeze_state()
        super(CfgNode, self).__setitem__(key, self._parse_value(value))

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if hasattr(self.__class__, key):
            raise AttributeError("attribute '{}' is read-only".format(key))
        self._check_freeze_state()
        self[key] = value

    def __setstate__(self, state):
        self._set_freeze_state(self, state['_frozen'])

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for k, v in self.items():
            other[deepcopy(k, memo)] = deepcopy(v, memo)
        return other

    def __eq__(self, other):
        if isinstance(other, Config):
            return self.__eq__(other._cfg)
        elif not isinstance(other, dict):
            return False
        elif list(self.keys()) != list(other.keys()):
            return False
        for key in self.keys():
            if self[key] != other[key]:
                return False
        return True

    def __repr__(self):
        return super(OrderedDict, self).__repr__()

    def _parse_value(self, value):
        if isinstance(value, dict):
            value = self.__class__(**value)
        elif isinstance(value, (list, tuple)):
            value = type(value)(self._parse_value(v) for v in value)
        elif isinstance(value, range):
            value = list(self._parse_value(v) for v in value)
        return value

    def _check_freeze_state(self):
        if self._frozen:
            raise RuntimeError('can not modify a frozen CfgNode')

    def freeze(self):
        self._set_freeze_state(self, True)

    def unfreeze(self):
        self._set_freeze_state(self, False)

    def copy(self):
        return deepcopy(self)

    def update(self, *args, **kwargs):
        other = dict()
        if len(args) == 1:
            other.update(args[0])
        elif len(args) > 1:
            raise TypeError('too many arguments')
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or (not isinstance(self[k], dict))
                    or (not isinstance(v, dict))):
                self[k] = self._parse_value(v)
            else:
                self[k].update(v)

    def to_dict(self, ordered=False):
        base = OrderedDict() if ordered else dict()
        for k, v in self.items():
            if isinstance(v, type(self)):
                base[k] = v.to_dict()
            elif isinstance(v, (list, tuple)):
                base[k] = type(v)(
                    o.to_dict() if isinstance(o, type(self)) else o for o in v)
            else:
                base[k] = v
        return base

    def to_json(self):
        return nncore.dumps(self.to_dict(), file_format='json', indent=2)


@bind_getter('filename')
class Config(object):
    """
    A facility for better :obj:`CfgNode` objects.

    This class is a wrapper for :obj:`CfgNode` which can be initialized from
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
            with TemporaryDirectory() as tmp_dir:
                copyfile(filename, osp.join(tmp_dir, '_tmp_config.py'))
                sys.path.insert(0, tmp_dir)
                mod = import_module('_tmp_config')
                sys.path.pop(0)
                cfg = {
                    k: v
                    for k, v in mod.__dict__.items()
                    if not k.startswith('__') or not k.endswith('__')
                }
        elif file_format in ['json', 'yml', 'yaml']:
            cfg = nncore.load(filename)
        else:
            raise TypeError("unsupported format: '{}'".format(file_format))

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
            super(Config, self).__setattr__('_text', None)

    @property
    def text(self):
        text = '' if self._filename is None else '* {} *\n\n'.format(
            self._filename)

        if self._text is not None:
            return text + self._text

        def _indent(attr_str):
            tokens = attr_str.split('\n')
            if len(tokens) == 1:
                return attr_str
            first = tokens.pop(0)
            tokens = [' ' * 4 + line for line in tokens]
            return '{}\n{}'.format(first, '\n'.join(tokens))

        def _basic(key, value):
            if isinstance(value, dict):
                v_str = _dict(value)
            elif isinstance(value, str):
                v_str = "'{}'".format(value)
            else:
                v_str = str(value)
            attr_str = v_str if key is None else '{} = {}'.format(key, v_str)
            return _indent(attr_str)

        def _iterable(key, value):
            tokens = []
            for v in value:
                if isinstance(v, dict):
                    tokens.append('dict({})'.format(_indent('\n' + _dict(v))))
                elif isinstance(v, (list, tuple)):
                    tokens.append(_iterable(None, v))
                else:
                    tokens.append(_basic(None, v))
            left, right = ('[', ']') if isinstance(value, list) else ('(', ')')
            v_str = '{}{}{}'.format(left, ', '.join(tokens), right)
            return v_str if key is None else '{} = {}'.format(key, v_str)

        def _dict(value, parent=False):
            tokens = []
            for idx, (k, v) in enumerate(value.items()):
                is_last = idx >= len(value) - 1
                end = '' if parent or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _dict(v)
                    attr_str = '{} = dict({}'.format(str(k), v_str)
                    attr_str = _indent(attr_str) + ')' + end
                elif isinstance(v, (list, tuple)):
                    attr_str = _iterable(k, v) + end
                else:
                    attr_str = _basic(k, v) + end
                tokens.append(attr_str)
            return '\n'.join(tokens)

        cfg_dict = self._cfg.to_dict()
        text += _dict(cfg_dict, parent=True)

        return text

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

    def __eq__(self, other):
        return self._cfg.__eq__(other)

    def __repr__(self):
        return 'Config(filename: {}): {}'.format(self._filename,
                                                 repr(self._cfg))

    def freeze(self):
        self._cfg.freeze()

    def unfreeze(self):
        self._cfg.unfreeze()

    def update(self, *args, **kwargs):
        args = [arg._cfg if isinstance(arg, Config) else arg for arg in args]
        self._cfg.update(*args, **kwargs)

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def setdefault(self, key, default):
        return self._cfg.setdefault(key, default)

    def copy(self):
        return self._cfg.copy()

    def pop(self, key, default):
        return self._cfg.pop(key, default)

    def to_dict(self, ordered=False):
        return self._cfg.to_dict(ordered=ordered)
