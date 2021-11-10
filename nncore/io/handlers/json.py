# Copyright (c) Ye Liu. All rights reserved.

import json

import jsonlines

from .base import FileHandler


class JSONHandler(FileHandler):
    """
    Handler for JSON files.
    """

    def load_from_file(self, file, **kwargs):
        return json.load(file, **kwargs)

    def dump_to_file(self, obj, file, **kwargs):
        json.dump(obj, file, **kwargs)

    def load_from_str(self, string, **kwargs):
        return json.loads(string, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return json.dumps(obj, **kwargs)


class JSONLHandler(FileHandler):
    """
    Handler for JSON Lines files.
    """

    def load_from_file(self, file):
        return [line for line in file]

    def dump_to_file(self, obj, file):
        if isinstance(obj, (list, tuple)):
            file.write_all(obj)
        else:
            file.write(obj)

    def load_from_path(self, path, mode='r'):
        with jsonlines.open(path, mode) as f:
            return self.load_from_file(f)

    def dump_to_path(self, obj, path, mode='w'):
        with jsonlines.open(path, mode) as f:
            self.dump_to_file(obj, f)
