Welcome to NNCore's documentation!
==================================

NNCore is a library that provides common functionalities for Machine Learning and Deep Learning researchers. This project aims at helping users focus more on science but not engineering during researches. The essential functionalities include but are not limited to:

- Universal I/O APIs
- Efficient implementations of layers and losses that are not included in PyTorch
- Extended methods for distributed training
- More powerful data loading techniques
- An engine that can take over the whole training and testing process, with all the baby-sitting works (stage control, optimizer configuration, lr scheduling, checkpoint management, metrics & tensorboard writing, etc.) done automatically.

.. toctree::
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   nncore.engine
   nncore.image
   nncore.io
   nncore.losses
   nncore.modules
   nncore.ops
   nncore.parallel
   nncore.utils
