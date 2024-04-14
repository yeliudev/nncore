<p align="center">
  <img src="https://raw.githubusercontent.com/yeliudev/nncore/main/.github/logo.svg">
</p>

<h1 align="center">NNCore</h1>

<p align="center">
  <strong>A lightweight machine learning toolkit for researchers.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/nncore">
    <img src="https://badgen.net/pypi/v/nncore?label=PyPI&cache=300">
  </a>
  <a href="https://pypistats.org/packages/nncore">
    <img src="https://badgen.net/pypi/dm/nncore?label=Downloads&color=cyan&cache=300">
  </a>
  <a href="https://github.com/yeliudev/nncore/blob/main/LICENSE">
    <img src="https://badgen.net/github/license/yeliudev/nncore?label=License&cache=300">
  </a>
  <a href="https://coveralls.io/github/yeliudev/nncore?branch=main">
    <img src="https://badgen.net/coveralls/c/github/yeliudev/nncore/main?label=Coverage&cache=300">
  </a>
  <a href="https://www.codacy.com/gh/yeliudev/nncore/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=yeliudev/nncore&amp;utm_campaign=Badge_Grade">
    <img src="https://badgen.net/codacy/grade/93d963e3247e43eb86d282cffacf0125?label=Code%20Quality&cache=300">
  </a>
</p>

NNCore is a library that provides common functionalities for Machine Learning and Deep Learning researchers. This project aims at helping users focus more on science but not engineering during research. The essential functionalities include but are not limited to:

- Universal I/O APIs
- Efficient implementations of layers and losses that are not included in PyTorch
- Extended methods for distributed training
- More powerful data loading techniques
- An engine that can take over the whole training and testing process, with all the baby-sitting works (stage control, optimizer configuration, lr scheduling, checkpoint management, metrics & tensorboard writing, etc.) done automatically. See an [example](https://github.com/yeliudev/nncore/blob/main/examples/mnist.py) for details.

Note that some methods in the library work with PyTorch 2.0+, but the installation of PyTorch is not necessary.

## Continuous Integration

| Platform / Python Version | 3.9 | 3.10 | 3.11 | 3.12
| :-: | :-: | :-: | :-: | :-: |
| Ubuntu 20.04 | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/0?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/1?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/2?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/3?icon=github&cache=300)][link] |
| Ubuntu 22.04 | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/4?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/5?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/6?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/7?icon=github&cache=300)][link] |
| macOS 12.x | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/8?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/9?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/10?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/11?icon=github&cache=300)][link] |
| macOS 13.x | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/12?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/13?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/14?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/15?icon=github&cache=300)][link] |
| Windows Server 2022 | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/16?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/17?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/18?icon=github&cache=300)][link] | [![Build](https://badgen.net/runkit/yeliudev/nncore-badge/19?icon=github&cache=300)][link] |

## Installation

You may install nncore directly from PyPI

```
pip install nncore
```

or manually from source

```
git clone https://github.com/yeliudev/nncore.git
cd nncore
pip install -e .
```

## Getting Started

Please refer to our [documentation](https://nncore.readthedocs.io/) for how to incorporate nncore into your projects.

## Acknowledgements

This library is licensed under the [MIT License](https://github.com/yeliudev/nncore/blob/main/LICENSE). Part of the code in this project is modified from [mmcv](https://github.com/open-mmlab/mmcv) and [fvcore](https://github.com/facebookresearch/fvcore) with many thanks to the original authors.

[link]: https://github.com/yeliudev/nncore/actions/workflows/build.yml
