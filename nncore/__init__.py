# Copyright (c) Ye Liu. All rights reserved.

import warnings

from .io import *  # noqa
from .utils import *  # noqa

try:
    from .image import *  # noqa
    from .video import *  # noqa
except ImportError:
    warnings.warn("Please install opencv-python to enable 'nncore.image'")

__version__ = '0.3.6'
