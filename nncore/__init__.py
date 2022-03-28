# Copyright (c) Ye Liu. All rights reserved.

from .io import *  # noqa
from .utils import *  # noqa

try:
    from .image import *  # noqa
    from .video import *  # noqa
except ImportError:
    from warnings import warn
    warn("Please install opencv-python to enable 'nncore.image' and "
         "'nncore.video'")

__version__ = '0.3.6'
