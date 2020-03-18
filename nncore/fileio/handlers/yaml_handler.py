# Copyright (c) Ye Liu. All rights reserved.

import yaml

from .base import FileHandler

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class YamlHandler(FileHandler):

    def load_from_fileobj(self, file, **kwargs):
        return yaml.load(file, Loader=Loader, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        yaml.dump(obj, file, Dumper=Dumper, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return yaml.dump(obj, Dumper=Dumper, **kwargs)
