# Copyright (c) Ye Liu. All rights reserved.

import json

from .base import FileHandler


class JsonHandler(FileHandler):

    def load_from_fileobj(self, file, **kwargs):
        return json.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        json.dump(obj, file, **kwargs)

    def load_from_str(self, string, **kwargs):
        return json.loads(string, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return json.dumps(obj, **kwargs)
