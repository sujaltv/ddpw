DDPW
####

The **Distributed Data Parallel wrapper** is a minimal PyTorch-based wrapper
that makes it easy to run deep learning models on a CPU, a GPU, multiple GPUSs,
or clustered nodes of GPUs. It also includes support for training Mac's Apple
M1.

This is a "DDP" wrapper; it supports data parallelism across processors, not
model parallelism.

Features
========

1. **Multiple platforms**. This wrapper allows models to be trained on:

   a. a single CPU,
   b. a single GPU,
   c. Mac's Apple M1 SoCs,
   d. multiple cluster-less GPUs, and
   e. SLURM-based multi-node (clustered) single/multiple GPUs (by using `Submitit <https://github.com/facebookincubator/submitit>`_)

2. **Flexible training and evaluation**. This wrapper allows easy training, pausing, resuming, and evaluating of models

.. toctree::
   :caption: Introduction
   :glob:
   :hidden:
   :titlesonly:

   quickstart/installation
   quickstart/example

.. toctree::
   :caption: API
   :glob:
   :hidden:
   :titlesonly:

   api/wrapper
   api/trainer
   api/optimiser
   api/types
   api/utils

.. toctree::
   :caption: Contribution
   :glob:
   :hidden:
   :titlesonly:

   contribution/source
   contribution/documentation

.. toctree::
   :caption: Miscellaneous
   :glob:
   :hidden:
   :titlesonly:

   quickstart/mnist

.. toctree::
   :caption: Bureau
   :glob:
   :hidden:
   :titlesonly:

   LICENCE

