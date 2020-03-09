# Copyright (c) Ye Liu. All rights reserved.

import sentry_sdk

from .fileio import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

__version__ = '0.0.1rc1'

sentry_sdk.init('https://59e084fdf48d40a995fd8d85621b9f58@sentry.io/4107977')
