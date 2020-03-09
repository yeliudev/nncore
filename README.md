<div align="center">

![Logo](.github/nncore-logo.svg)

# NNCore

**A lightweight PyTorch code wrapper for ML researchers.**

[![PyPI version](https://badge.fury.io/py/nncore.svg)](https://badge.fury.io/py/nncore)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/c1aris/nncore/badge.svg?branch=master)](https://coveralls.io/github/c1aris/nncore?branch=master)
[![Slack](https://img.shields.io/badge/slack-chat-blue.svg?logo=slack)](https://join.slack.com/t/nncore/shared_invite/zt-cex52vw2-PBxlf~BToxS3k8etdxYxHQ)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/61c2ce72b14345d186876150cc1d4df8)](https://www.codacy.com/manual/c1aris/nncore?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=c1aris/nncore&amp;utm_campaign=Badge_Grade)

</div>

NNCore is a core library that provides common functionalities for Machine Learning and Deep Learning researchers. This project aims at helping people focus more on science but not engineering during researches. The essential functionalities include but are not limited to:

* Universal I/O APIs
* Efficient implementations of layers and losses that are not included in PyTorch
* Extended methods for distributed trainings
* Useful utilities
* **TODO**: A core engine that can take over the whole training and testing process, with all the baby-sitting works (optimizer configuration, checkpoint management, log file & Tensorboard writing, stage control, etc.) done automatically.

Note that some methods in the library work with PyTorch 1.3+, but the installation of PyTorch is not necessary.

## Continuous Integration

| Platform / Python Version | 3.6 | 3.7 | 3.8 | nightly | pypy3 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Ubuntu 16.04 | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/1/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/2/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/3/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/4/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/5/com)](https://travis-ci.com/c1aris/nncore) |
| Ubuntu 18.04 | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/6/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/7/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/8/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/9/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/10/com)](https://travis-ci.com/c1aris/nncore) |
| macOS 10.13 | <center>—</center> | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/11/com)](https://travis-ci.com/c1aris/nncore) | <center>—</center> | <center>—</center> | <center>—</center> |
| macOS 10.14 | <center>—</center> | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/12/com)](https://travis-ci.com/c1aris/nncore) | <center>—</center> | <center>—</center> | <center>—</center> |
| Windows | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/13/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/14/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/15/com)](https://travis-ci.com/c1aris/nncore) | [![Build Status](https://catcatserver.xyz/badge/c1aris/nncore/master/16/com)](https://travis-ci.com/c1aris/nncore) | <center>—</center> |

## Installation

You may install nncore from PyPI

```
pip install nncore
```

or directly from source

```
git clone https://github.com/c1aris/nncore.git
cd nncore
pip install -e .
```

## Acknowledgements

This library is licensed under the [MIT License](LICENSE). Part of the code in this project is modified from [mmcv](https://github.com/open-mmlab/mmcv) and [fvcore](https://github.com/facebookresearch/fvcore) with many thanks to the original authors.
