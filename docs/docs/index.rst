DDPW
####

**Distributed Data Parallel Wrapper (DDPW)** is a minimal PyTorch-based wrapper
that makes it easy to run deep learning models on various compute platforms.
DDPW handles scaffolding tasks like creating threads on GPUs/nodes, moving the
models and optimisers to devices, setting up inter-process communication,
`etc.`, and allows the user to focus on the main aspects of modelling and
training.

DDPW includes support for training on CPUs, GPUs (CUDA and Apple M1 SoC) and in
the SLURM environment.

Features
========

1. **Multiple platforms**. This wrapper allows models to be trained different
   compute platforms:

   a. CPU,
   b. one or more GPUs (CUDA and Apple M1 SoC), and
   c. SLURM (by using `Submitit
      <https://github.com/facebookincubator/submitit>`_)

2. **Flexible training and evaluation**. This wrapper allows easy training,
   pausing, resuming, and evaluating of models.

3. **Non-training tasks**. Tasks that are not essentially of the model-based or
   training-based paradigms but those that simply need to be run across devices
   can also be executed.

Example
=======

.. code-block:: python
    :linenos:
    :emphasize-lines: 13,16,20,24,25

    from ddpw.platform import Platform, PlatformConfig
    from ddpw.artefacts import ArtefactsConfig
    from ddpw.job import JobConfig, JobMode
    from ddpw.wrapper import Wrapper

    from src import MyDataset, MyModel, MyTrainer, my_optimiser

    # datasets
    train_set = MyDataset(train=True)
    test_set = MyDataset(train=False)

    # configure the platform
    p_config = PlatformConfig(platform=Platform.GPU, n_gpus=4, cpus_per_task=2)

    # configure the artefacts (model, dataset, optimiser, etc.)
    a_config = ArtefactsConfig(train_set=train_set, test_set=test_set,
        batch_size=64, model=MyModel(), optimiser_loader=my_optimiser)

    # configure the job
    j_config = JobConfig(job_type=JobMode.TRAIN, start_at=0, epochs=50,
        save_every=5)

    # call the job
    w = Wrapper(p_config, a_config)
    w.start(MyTrainer(j_config))

Refer to the :ref:`example with MNIST <MNIST example>` to see how the custom
artefacts are implemented.

.. toctree::
   :caption: Introduction
   :glob:
   :hidden:
   :titlesonly:

   quickstart/installation

.. toctree::
   :caption: API
   :glob:
   :hidden:
   :titlesonly:

   api/wrapper
   api/job
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
