# Copyright (c) Ye Liu. Licensed under the MIT License.

import logging
import tempfile
from platform import system

import nncore


def test_get_logger():
    logger1 = nncore.get_logger('test1')
    assert isinstance(logger1, logging.Logger)
    assert logger1.level == logging.INFO
    assert len(logger1.handlers) == 1
    assert isinstance(logger1.handlers[0], logging.StreamHandler)

    logger2 = nncore.get_logger('test2', log_level=logging.DEBUG)
    assert isinstance(logger2, logging.Logger)
    assert logger2.level == logging.DEBUG
    assert len(logger2.handlers) == 1

    if system() != 'Windows':
        with tempfile.NamedTemporaryFile() as f:
            logger3 = nncore.get_logger('test3', log_file=f.name)
        assert isinstance(logger3, logging.Logger)
        assert len(logger3.handlers) == 2
        assert isinstance(logger3.handlers[0], logging.StreamHandler)
        assert isinstance(logger3.handlers[1], logging.FileHandler)

    logger4 = nncore.get_logger('test2')
    assert id(logger4) == id(logger2)

    logger5 = nncore.get_logger('test5')
    assert logger5.handlers == logger5.handlers
