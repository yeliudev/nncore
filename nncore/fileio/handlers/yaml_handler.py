# Copyright (c) Ye Liu. All rights reserved.

import yaml

from .base import FileHandler

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class YamlHandler(FileHandler):

    def load_from_fileobj(self, file):
        return yaml.load(file, Loader=Loader)

    def dump_to_fileobj(self, obj, file):
        yaml.dump(obj, file, Dumper=Dumper)

    def dump_to_bytes(self, obj):
        return yaml.dump(obj, Dumper=Dumper)
