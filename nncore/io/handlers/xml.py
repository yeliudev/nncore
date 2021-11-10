# Copyright (c) Ye Liu. All rights reserved.

import xml.etree.ElementTree as ET

from .base import FileHandler


class XMLHandler(FileHandler):
    """
    Handler for XML files.
    """

    def load_from_file(self, file, **kwargs):
        tree = ET.parse(file, **kwargs)
        return tree.getroot()

    def dump_to_file(self, obj, file, **kwargs):
        tree = ET.ElementTree(obj)
        tree.write(file, **kwargs)

    def load_from_str(self, string, **kwargs):
        return ET.fromstring(string, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return ET.tostring(obj, **kwargs)

    def load_from_path(self, path, **kwargs):
        return self.load_from_file(path, **kwargs)

    def dump_to_path(self, obj, path, **kwargs):
        self.dump_to_file(obj, path, **kwargs)
