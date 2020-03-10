# Copyright (c) Ye Liu. All rights reserved.

import time

import pytest

import nncore


def test_timer_init():
    timer = nncore.Timer(start=False)
    assert not timer.is_running
    timer.start()
    assert timer.is_running
    timer = nncore.Timer()
    assert timer.is_running


def test_timer_run():
    timer = nncore.Timer()
    time.sleep(1)
    assert abs(timer.since_start() - 1) < 1
    time.sleep(1)
    assert abs(timer.since_last_check() - 1) < 1
    assert abs(timer.since_start() - 2) < 1
    timer = nncore.Timer(False)
    with pytest.raises(RuntimeError):
        timer.since_start()
    with pytest.raises(RuntimeError):
        timer.since_last_check()


def test_timer_context(capsys):
    with nncore.Timer():
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert abs(float(out) - 1) < 1
    with nncore.Timer(print_tmpl='time: {:.1f}s'):
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert out == 'time: 1.0s\n' or out == 'time: 1.1s\n'
