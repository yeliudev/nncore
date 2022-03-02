# Copyright (c) Ye Liu. All rights reserved.

import os
import sys
from collections import OrderedDict
from copy import deepcopy
from importlib import import_module
from tempfile import TemporaryDirectory

import nncore
from .binder import bind_getter
from .data import is_seq_of
from .path import abs_path, cp, dir_name, is_file, join, pure_ext


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
        for key, value in kwargs.items():
            self[key] = value

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

        def _copy(obj, memo):
            try:
                return deepcopy(obj, memo)
            except TypeError:
                return obj

        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            key = _copy(key, memo)
            if isinstance(value, (list, tuple)):
                value = type(value)(_copy(v, memo) for v in value)
            else:
                value = _copy(value, memo)
            other[key] = value
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
        return value

    def _check_freeze_state(self):
        if self._frozen:
            raise RuntimeError('can not modify a frozen {} object'.format(
                self.__class__))

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
        for key, value in other.items():
            if key not in self or not isinstance(
                    self[key], dict) or not isinstance(value, dict):
                self[key] = self._parse_value(value)
            else:
                self[key].update(value)

    def merge_from(self, other):

        def _insert(ori, value, cfg):
            assert isinstance(cfg, (dict, int))
            if isinstance(cfg, dict):
                assert is_seq_of(cfg['index'], int) or isinstance(
                    cfg['index'], int)
                if isinstance(cfg['index'], (list, tuple)):
                    assert isinstance(cfg['value'], (list, tuple)) and len(
                        cfg['index']) == len(cfg['value'])
                    idxs, vals = cfg['index'], cfg['value']
                else:
                    idxs, vals = [cfg['index']], [cfg['value']]
                for i, idx in enumerate(idxs):
                    ori.insert(idx, vals[i])
            elif type(cfg) is int:
                ori.insert(cfg, value)
            else:
                ori.append(value)
            return ori

        def _update(ori, value, cfg):
            assert isinstance(cfg, (dict, int))
            if isinstance(cfg, dict):
                assert is_seq_of(cfg['index'], int) or isinstance(
                    cfg['index'], int)
                if isinstance(cfg['index'], (list, tuple)):
                    assert isinstance(cfg['value'], (list, tuple)) and len(
                        cfg['index']) == len(cfg['value'])
                    idxs, vals = cfg['index'], cfg['value']
                else:
                    idxs, vals = [cfg['index']], [cfg['value']]
                for i, idx in enumerate(idxs):
                    if isinstance(ori[idx], dict) and isinstance(
                            vals[i], dict):
                        ori[idx].merge_from(vals[i])
                    else:
                        ori[idx] = vals[i]
            elif isinstance(ori[cfg], dict):
                ori[cfg].merge_from(value)
            else:
                ori[cfg] = value
            return ori

        assert isinstance(other, dict)
        other = self.__class__(**other)

        for key, value in other.items():
            if key.startswith('_'):
                continue

            if isinstance(value, dict):
                refine = value.pop('_refine_', None)
                repeat = value.pop('_repeat_', None)
                delete = value.pop('_delete_', None)
                insert = value.pop('_insert_', None)
                update = value.pop('_update_', None)

                do_insert = type(insert) is int or insert
                do_update = type(update) is int or update

                if key not in self and refine:
                    continue

                if key in self and isinstance(repeat, int):
                    self[key] = [self[key] for _ in range(repeat)]
                    continue

                if key in self and isinstance(self[key], dict):
                    assert not do_update and not (delete and do_insert)
                    if do_insert:
                        self[key] = _insert([self[key]], value, insert)
                        continue
                    elif not delete:
                        self[key].merge_from(value)
                        continue

                if key in self and isinstance(self[key], (list, tuple)):
                    if do_insert or do_update:
                        is_tuple = isinstance(self[key], tuple)
                        if is_tuple:
                            self[key] = list(self[key])
                        if do_insert:
                            self[key] = _insert(self[key], value, insert)
                        if do_update:
                            self[key] = _update(self[key], value, update)
                        if is_tuple:
                            self[key] = tuple(self[key])
                        continue

                assert not do_insert and not do_update
                tmp_node = self.__class__()
                tmp_node.merge_from(value)
                if tmp_node or not value:
                    self[key] = tmp_node
            elif value == '_delete_':
                if key in self:
                    del self[key]
            else:
                self[key] = value

    def to_dict(self, ordered=False):
        base = OrderedDict() if ordered else dict()
        for key, value in self.items():
            if isinstance(value, self.__class__):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    v.to_dict() if isinstance(v, self.__class__) else v
                    for v in value)
            else:
                base[key] = value
        return base

    def to_json(self, indent=2):
        return nncore.dumps(self.to_dict(), format='json', indent=indent)


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
                formats include ``py``, ``json``, and ``yaml/yml``.
            freeze (bool, optional): Whether to freeze the config after
                initialization.  Default: ``False``.

        Returns:
            :obj:`Config`: The constructed config object.
        """
        filename = abs_path(filename)
        is_file(filename, raise_error=True)

        format = pure_ext(filename)
        if format == 'py':
            with TemporaryDirectory() as tmp:
                mod_name = str(int.from_bytes(os.urandom(2), 'big'))
                cp(filename, join(tmp, '{}.py'.format(mod_name)))
                sys.path.insert(0, tmp)
                mod = import_module(mod_name)
                sys.path.pop(0)
                cfg = {
                    k: v
                    for k, v in mod.__dict__.items()
                    if not k.startswith('__') or not k.endswith('__')
                }
        elif format in ('json', 'yml', 'yaml'):
            cfg = nncore.load(filename)
        else:
            raise TypeError("unsupported format: '{}'".format(format))

        if '_base_' in cfg:
            base = cfg.pop('_base_')
            if isinstance(base, str):
                base = [base]

            _cfg = CfgNode()
            for name in base:
                if name.endswith(('.py', '.json', '.yml', '.yaml')):
                    path = join(dir_name(filename), name)
                    _cfg.merge_from(Config.from_file(path))
                else:
                    curr_path = os.getcwd()
                    if curr_path not in sys.path:
                        sys.path.insert(0, curr_path)
                    import_module(name)

            _cfg.merge_from(cfg)
            cfg = _cfg

        return Config(cfg, filename=filename, freeze=freeze)

    def __init__(self, *args, filename=None, freeze=False, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        super(CfgNode, self).__setattr__('_filename', filename)

        if freeze:
            self.freeze()

    def __repr__(self):
        return '{}({}frozen={}): {}'.format(
            self.__class__.__name__, '' if self._filename is None else
            "filename='{}', ".format(self._filename), self._frozen,
            super(Config, self).__repr__())

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
            prefix = '\n' if len(value) > 1 else ''
            expand = any(isinstance(v, (dict, list, tuple)) for v in value)
            for v in value:
                if isinstance(v, dict):
                    if len(v) > 1:
                        a_str = _indent('\n' + _dict(v))
                    else:
                        a_str = _dict(v)
                    tokens.append(_indent(prefix + 'dict({})'.format(a_str)))
                elif isinstance(v, (list, tuple)):
                    tokens.append(_indent(prefix + _iterable(None, v)))
                else:
                    a_str = _basic(None, v)
                    tokens.append(_indent(prefix + a_str) if expand else a_str)
            left, right = ('[', ']') if isinstance(value, list) else ('(', ')')
            sep = ',' if expand else ', '
            v_str = '{}{}{}'.format(left, sep.join(tokens), right)
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
