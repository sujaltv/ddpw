DDPW
####

The **Distributed Data Parallel wrapper** is a minimal PyTorch-based wrapper
written in Python 3.8 that makes it easy to run deep learning models on multiple
nodes and GPUs as well as on a single CPU.

This is a "DDP" wrapper; it supports data parallelism across GPUs, not model
parallelism.

Features
========

1. **Multiple platforms**. This wrapper allows models to be trained on:

   a. a single CPU,
   b. a single GPU,
   c. multiple cluster-less GPUs, and
   d. SLURM-based multi-node (clustered) single/multiple GPUs (by using `Submitit <https://github.com/facebookincubator/submitit>`_)

2. **Logging**. This wrapper includes APIs for logging with Tensorboard
3. **Flexible training and evaluation**. This wrapper allows easy training, pausing, resuming, and evaluation. This means that one could train a model for :math:`50` epochs, and later on resume the training from the :math:`50^\textrm{th}` epoch

Index
=====

.. toctree::
   :caption: Introduction
   :glob:
   :titlesonly:

   quickstart/installation
   quickstart/examples

.. toctree::
   :caption: API
   :glob:
   :titlesonly:

   api/ddpwrapper
   api/platform
   api/trainer
   api/logger

.. toctree::
   :caption: Contribution
   :glob:
   :titlesonly:

   contribution/setup
   contribution/documentation
