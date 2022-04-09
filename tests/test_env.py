# Copyright (c) Ye Liu. Licensed under the MIT License.

import nncore


def test_get_host_info():
    assert '@' in nncore.get_host_info()


def test_collect_env_info():
    assert 'nncore' in nncore.collect_env_info()
