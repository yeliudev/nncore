# Copyright (c) Ye Liu. All rights reserved.

from io import StringIO

import yaml

from .base import FileHandler

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


class YamlHandler(FileHandler):
    """
    Handler for YAML files.
    """

    def load_from_file(self, file, **kwargs):
        return yaml.load(file, Loader=Loader, **kwargs)

    def dump_to_file(self, obj, file, **kwargs):
        yaml.dump(obj, file, Dumper=Dumper, **kwargs)

    def load_from_str(self, string, **kwargs):
        return yaml.load(string, Loader=Loader, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        io = StringIO()
        yaml.dump(obj, io, Dumper=Dumper, **kwargs)
        return io.getvalue()
