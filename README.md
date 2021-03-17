<div align="center">

![Logo](https://github.com/yeliudev/nncore/blob/master/.github/nncore-logo.svg)

# NNCore

**A lightweight PyTorch code wrapper for ML researchers.**

[![PyPI version](https://badge.fury.io/py/nncore.svg)](https://pypi.org/project/nncore/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/yeliudev/nncore/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/yeliudev/nncore/badge.svg?branch=master)](https://coveralls.io/github/yeliudev/nncore?branch=master)
[![Slack](https://img.shields.io/badge/slack-chat-blue.svg?logo=slack)](https://join.slack.com/t/nncore/shared_invite/zt-cex52vw2-PBxlf~BToxS3k8etdxYxHQ)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/0692961de1d94464a770b22efc2a5b0d)](https://www.codacy.com/manual/yeliudev/nncore?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=yeliudev/nncore&amp;utm_campaign=Badge_Grade)

</div>

NNCore is a library that provides common functionalities for Machine Learning and Deep Learning researchers. This project aims at helping users focus more on science but not engineering during researches. The essential functionalities include but are not limited to:

* Universal I/O APIs
* Efficient implementations of layers and losses that are not included in PyTorch
* Extended methods for distributed training
* More powerful data loading techniques
* An engine that can take over the whole training and testing process, with all the baby-sitting works (stage control, optimizer configuration, lr scheduling, checkpoint management, metrics & tensorboard writing, etc.) done automatically. See an [example](https://github.com/yeliudev/nncore/blob/master/examples/mnist.py) for details.

Note that some methods in the library work with PyTorch 1.6+, but the installation of PyTorch is not necessary.

## Continuous Integration

[ci_root]: https://travis-ci.com/yeliudev/nncore

| Platform / Python Version | 3.6 | 3.7 | 3.8 | 3.9 |
| :-: | :-: | :-: | :-: | :-: |
| Ubuntu 16.04 | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/1)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/2)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/3)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/4)][ci_root] |
| Ubuntu 18.04 | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/5)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/6)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/7)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/8)][ci_root] |
| Ubuntu 20.04 | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/9)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/10)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/11)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/12)][ci_root] |
| macOS 10.14 | — | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/13)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/14)][ci_root] | — |
| macOS 10.15 | — | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/15)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/16)][ci_root] | — |
| Windows | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/17)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/18)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/19)][ci_root] | [![Build Status](https://api.catcatserver.xyz/badge/yeliudev/nncore/master/20)][ci_root] |

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

This library is licensed under the [MIT License](https://github.com/yeliudev/nncore/blob/master/LICENSE). Part of the code in this project is modified from [mmcv](https://github.com/open-mmlab/mmcv) and [fvcore](https://github.com/facebookresearch/fvcore) with many thanks to the original authors.
