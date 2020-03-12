# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp
import sys
from importlib import import_module

from addict import Dict

import nncore


class _ConfigDict(Dict):

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, key):
        try:
            value = super(_ConfigDict, self).__getattr__(key)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, key))
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config(object):
    """
    A facility for config and config files.

    It supports common file formats as configs: py/json/yaml. The interface is
    the same as a dict object and also allows access config values as
    attributes.
    """

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))

        super(Config, self).__setattr__('_cfg_dict', _ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)

        if filename:
            with open(filename, 'r') as f:
                super(Config, self).__setattr__('_text', f.read())
        else:
            super(Config, self).__setattr__('_text', '')

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = _ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = _ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)


def build_config(filename):
    """
    Create a Config object from a config file.

    Args:
        filename (str): name of the config file

    Returns:
        cfg (Config): the created Config object
    """
    filename = osp.abspath(osp.expanduser(filename))
    nncore.file_exist(filename, raise_error=True)

    file_format = filename.split('.')[-1]
    if file_format == 'py':
        module_name = osp.basename(filename)[:-3]
        config_dir = osp.dirname(filename)

        if '.' in module_name:
            raise ValueError('dots are not allowed in the config file path.')

        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)

        cfg_dict = {
            k: v
            for k, v in mod.__dict__.items() if not k.startswith('__')
        }
    elif file_format in ['yml', 'yaml', 'json']:
        cfg_dict = nncore.load(filename)
    else:
        raise TypeError('unsupported format: {}'.format(file_format))

    return Config(cfg_dict=cfg_dict, filename=filename)
