<div align="center">

![Logo](https://raw.githubusercontent.com/yeliudev/nncore/main/.github/nncore-logo.svg)

# NNCore

**A lightweight PyTorch code wrapper for ML researchers.**

[![PyPI Version](https://badge.fury.io/py/nncore.svg)](https://pypi.org/project/nncore/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/yeliudev/nncore/blob/main/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/yeliudev/nncore/badge.svg?branch=main)](https://coveralls.io/github/yeliudev/nncore?branch=main)
[![Slack](https://img.shields.io/badge/slack-chat-blue.svg?logo=slack)](https://join.slack.com/t/nncore-dev/shared_invite/zt-nr4fnk5j-qQoeUo38KBzgwVUdU_Wk8w)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/c55266f7dc904b5d8c31b15bafb1117c)](https://www.codacy.com/gh/yeliudev/nncore/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=yeliudev/nncore&amp;utm_campaign=Badge_Grade)

</div>

NNCore is a library that provides common functionalities for Machine Learning and Deep Learning researchers. This project aims at helping users focus more on science but not engineering during researches. The essential functionalities include but are not limited to:

* Universal I/O APIs
* Efficient implementations of layers and losses that are not included in PyTorch
* Extended methods for distributed training
* More powerful data loading techniques
* An engine that can take over the whole training and testing process, with all the baby-sitting works (stage control, optimizer configuration, lr scheduling, checkpoint management, metrics & tensorboard writing, etc.) done automatically. See an [example](https://github.com/yeliudev/nncore/blob/main/examples/mnist.py) for details.

Note that some methods in the library work with PyTorch 1.6+, but the installation of PyTorch is not necessary.

## Continuous Integration

| Platform / Python Version | 3.6 | 3.7 | 3.8 | 3.9 |
| :-: | :-: | :-: | :-: | :-: |
| Ubuntu 16.04 | [![build]][link] | [![build]][link] | [![build]][link] | [![build]][link] |
| Ubuntu 18.04 | [![build]][link] | [![build]][link] | [![build]][link] | [![build]][link] |
| Ubuntu 20.04 | [![build]][link] | [![build]][link] | [![build]][link] | [![build]][link] |
| macOS 10.15 | [![build]][link] | [![build]][link] | [![build]][link] | [![build]][link] |
| Windows | [![build]][link] | [![build]][link] | [![build]][link] | [![build]][link] |

## Installation

You may install nncore from PyPI

```
pip install nncore
```

or directly from source

```
git clone https://github.com/yeliudev/nncore.git
cd nncore
pip install -e .
```

## Acknowledgements

This library is licensed under the [MIT License](https://github.com/yeliudev/nncore/blob/main/LICENSE). Part of the code in this project is modified from [mmcv](https://github.com/open-mmlab/mmcv) and [fvcore](https://github.com/facebookresearch/fvcore) with many thanks to the original authors.

[build]: https://github.com/yeliudev/nncore/actions/workflows/build.yml/badge.svg
[link]: https://github.com/yeliudev/nncore/actions/workflows/build.yml
