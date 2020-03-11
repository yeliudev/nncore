# Copyright (c) Ye Liu. All rights reserved.

import multiprocessing as mp
import os
import time
from io import StringIO
from platform import system
from unittest.mock import patch

import nncore


def reset_string_io(io):
    io.truncate(0)
    io.seek(0)


class TestProgressBar(object):

    def test_start(self):
        out = StringIO()
        # without total task num
        nncore.ProgressBar(stream=out)
        assert out.getvalue() == 'completed: 0, elapsed: 0s'
        reset_string_io(out)
        # with total task num
        nncore.ProgressBar(10, stream=out)
        assert out.getvalue() == '[{}] 0/10, elapsed: 0s, ETA:'.format(' ' *
                                                                       50)
        reset_string_io(out)

    def test_update(self):
        out = StringIO()
        # without total task num
        prog_bar = nncore.ProgressBar(stream=out)
        time.sleep(1)
        reset_string_io(out)
        prog_bar.update()
        # enable robustness to slow CI
        assert out.getvalue(
        ) == 'completed: 1, elapsed: 1s, 1.0 tasks/s' or out.getvalue(
        ) == 'completed: 1, elapsed: 1s, 0.9 tasks/s'
        reset_string_io(out)
        # with total task num
        prog_bar = nncore.ProgressBar(10, stream=out)
        time.sleep(1)
        reset_string_io(out)
        prog_bar.update()
        # enable robustness to slow CI
        assert out.getvalue() == (
            '\r[{}] 1/10, 1.0 task/s, '
            'elapsed: 1s, ETA:     9s'.format('>' * 3 + ' ' * 31)
        ) or out.getvalue() == ('\r[{}] 1/10, 0.9 task/s, '
                                'elapsed: 1s, ETA:    10s'.format('>' * 3 +
                                                                  ' ' * 31))

    @patch.dict('os.environ', {'COLUMNS': '80'})
    def test_adaptive_length(self):
        out = StringIO()
        prog_bar = nncore.ProgressBar(10, stream=out)
        time.sleep(1)
        reset_string_io(out)
        prog_bar.update()
        assert len(out.getvalue()) == 80

        os.environ['COLUMNS'] = '30'
        reset_string_io(out)
        prog_bar.update()
        assert len(out.getvalue()) == 48

        os.environ['COLUMNS'] = '60'
        reset_string_io(out)
        prog_bar.update()
        assert len(out.getvalue()) == 60


def dummy_func(num):
    time.sleep(0.1)
    return num


def test_track_progress_list():
    ret = nncore.track_progress(dummy_func, [1, 2, 3])
    assert ret == [1, 2, 3]


def test_track_progress_iterator():
    ret = nncore.track_progress(dummy_func, ((i for i in [1, 2, 3]), 3))
    assert ret == [1, 2, 3]


def test_track_iter_progress():
    ret = []
    for num in nncore.track_iter_progress([1, 2, 3]):
        ret.append(dummy_func(num))
    assert ret == [1, 2, 3]


def test_track_enum_progress():
    ret = []
    count = []
    for i, num in enumerate(nncore.track_iter_progress([1, 2, 3])):
        ret.append(dummy_func(num))
        count.append(i)
    assert ret == [1, 2, 3]
    assert count == [0, 1, 2]


def test_track_parallel_progress_list():
    if system() == 'Windows':
        return
    mp.set_start_method('fork', force=True)
    results = nncore.track_parallel_progress(dummy_func, [1, 2, 3, 4], 2)
    assert results == [1, 2, 3, 4]


def test_track_parallel_progress_iterator():
    if system() == 'Windows':
        return
    mp.set_start_method('fork', force=True)
    results = nncore.track_parallel_progress(dummy_func,
                                             ((i for i in [1, 2, 3, 4]), 4), 2)
    assert results == [1, 2, 3, 4]
