# Copyright (c) Ye Liu. All rights reserved.

import json

from .base import FileHandler


class JsonHandler(FileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file):
        json.dump(obj, file)

    def dump_to_bytes(self, obj):
        return json.dumps(obj)
