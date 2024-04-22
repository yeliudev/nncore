Welcome to NNCore's documentation!
==================================

NNCore is a library that provides common functionalities for Machine Learning and Deep Learning researchers. This project aims at helping users focus more on science but not engineering during research. The essential functionalities include but are not limited to:

- Universal I/O APIs
- Efficient implementations of layers and losses that are not included in PyTorch
- Extended methods for distributed training
- More powerful data loading techniques
- An engine that can take over the whole training and testing process, with all the baby-sitting works (stage control, optimizer configuration, lr scheduling, checkpoint management, metrics & tensorboard writing, etc.) done automatically.

Note that some methods in the library work with PyTorch 2.0+, but the installation of PyTorch is not necessary.

.. toctree::
   :caption: Getting Started

   getting_started

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   nncore.dataset
   nncore.engine
   nncore.image
   nncore.io
   nncore.nn
   nncore.ops
   nncore.optim
   nncore.parallel
   nncore.utils
   nncore.video
