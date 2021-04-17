# Copyright (c) Ye Liu. All rights reserved.

import os
import sys
from collections import OrderedDict
from copy import deepcopy
from importlib import import_module
from tempfile import TemporaryDirectory

import nncore
from .binder import bind_getter


class CfgNode(OrderedDict):
    """
    An extended :obj:`OrderedDict` class with several practical methods.

    This class is an extension of the built-in type :obj:`OrderedDict`. The
    interface is the same as a dict object and also allows access config values
    as attributes. The input to the init method can be either a single dict or
    several named parameters.
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
            if isinstance(args[0], dict):
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
            raise RuntimeError('can not modify a frozen CfgNode object')

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
        return nncore.dumps(self.to_dict(), format='json', indent=2)


@bind_getter('filename')
class Config(CfgNode):
    """
    A facility for better :obj:`CfgNode` objects.

    This class inherits from :obj:`CfgNode` and it can be initialized from a
    config file. Users can use the static method :obj:`Config.from_file` to
    create a :obj:`Config` object.
    """

    @staticmethod
    def from_file(filename, freeze=False):
        """
        Build a :obj:`Config` object from a file.

        Args:
            filename (str): Path to the config file. Currently supported
                formats include ``py``, ``json`` and ``yaml/yml``.
            freeze (bool, optional): Whether to freeze the config after
                initialization.  Default: ``False``.

        Returns:
            :obj:`Config`: The constructed config object.
        """
        filename = nncore.abs_path(filename)
        nncore.is_file(filename, raise_error=True)

        format = nncore.pure_ext(filename)
        if format == 'py':
            with TemporaryDirectory() as tmp:
                mod_name = str(int.from_bytes(os.urandom(2), 'big'))
                nncore.cp(filename, nncore.join(tmp, '{}.py'.format(mod_name)))
                sys.path.insert(0, tmp)
                mod = import_module(mod_name)
                sys.path.pop(0)
                cfg = {
                    k: v
                    for k, v in mod.__dict__.items()
                    if not k.startswith('__') or not k.endswith('__')
                }
        elif format in ['json', 'yml', 'yaml']:
            cfg = nncore.load(filename)
        else:
            raise TypeError("unsupported format: '{}'".format(format))

        return Config(cfg, filename=filename, freeze=freeze)

    def __init__(self, *args, filename=None, freeze=False, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        super(CfgNode, self).__setattr__('_filename', filename)

        if freeze:
            self.freeze()

    @property
    def text(self):

        def _indent(a_str):
            tokens = a_str.split('\n')
            if len(tokens) == 1:
                return a_str
            first = tokens.pop(0)
            tokens = [' ' * 4 + line for line in tokens]
            return '{}\n{}'.format(first, '\n'.join(tokens))

        def _basic(key, value, blank=True):
            base_str = '{} = {}' if blank else '{}={}'
            if isinstance(value, dict):
                v_str = _dict(value)
            elif isinstance(value, str):
                v_str = "'{}'".format(value)
            else:
                v_str = str(value)
            a_str = v_str if key is None else base_str.format(key, v_str)
            return _indent(a_str)

        def _iterable(key, value, blank=True):
            base_str, tokens = '{} = {}' if blank else '{}={}', []
            expand = any(isinstance(v, (dict, list, tuple)) for v in value)
            for v in value:
                if isinstance(v, dict):
                    if len(v) > 1:
                        a_str = '\ndict({})'.format(_indent('\n' + _dict(v)))
                    else:
                        a_str = '\ndict({})'.format(_dict(v))
                    tokens.append(_indent(a_str))
                elif isinstance(v, (list, tuple)):
                    tokens.append(_indent('\n' + _iterable(None, v)))
                else:
                    a_str = _basic(None, v)
                    tokens.append(_indent('\n' + a_str) if expand else a_str)
            left, right = ('[', ']') if isinstance(value, list) else ('(', ')')
            v_str = '{}{}{}'.format(left, ', '.join(tokens), right)
            return v_str if key is None else base_str.format(key, v_str)

        def _dict(value, parent=False):
            base_str, tokens = '{} = dict({})' if parent else '{}=dict({})', []
            for i, (k, v) in enumerate(value.items()):
                end = '' if parent or i >= len(value) - 1 else ','
                if isinstance(v, dict):
                    if len(v) > 1:
                        a_str = base_str.format(str(k), '\n' + _dict(v))
                        a_str = _indent(a_str) + end
                    else:
                        a_str = base_str.format(str(k), _dict(v)) + end
                elif isinstance(v, (list, tuple)):
                    a_str = _iterable(k, v, blank=parent) + end
                else:
                    a_str = _basic(k, v, blank=parent) + end
                tokens.append(a_str)
            return '\n'.join(tokens)

        text = _dict(self.to_dict(), parent=True)
        if self._filename is not None:
            text = '{}\n'.format(self._filename) + text

        return text

    def __repr__(self):
        return '{}({}frozen={}): {}'.format(
            self.__class__.__name__, '' if self._filename is None else
            "filename='{}', ".format(self._filename), self._frozen,
            super(Config, self).__repr__())
