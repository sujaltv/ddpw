DDP Wrapper
###########

This is a minimal PyTorch-based wrapper that makes it easy to run deep learning
models on multiple nodes and GPUs as well as on a single CPU. This wrapper
allows the models to be trained on:

1. a single CPU,
2. a cluster-less GPU,
3. multiple cluster-less GPUs, and
4. SLURM-based multi-node (clustered) multiple GPUs (by using
   `Submitit <https://github.com/facebookincubator/submitit>`_)

It also includes APIs for logging with Tensorboard and saving and loading
checkpoints.

This is a "DDP" wrapper; it supports data parallelism across GPUs, not model
parallelism.

API
===

Training on a CPU
-----------------

While training on a single CPU requires no additional code such as this wrapper,
it can however be used to train on a single CPU. The wrapper virtually does
nothing but run the training iterations provided. This feature is included only
to conveniently switch training between CPU and GPUs merely by specifying
different CLI arguments.

.. code-block:: python
    :linenos:

    from ddpwrapper import DDPWrapper, Platform, Trainer

    class CustomTrainer(Trainer):
      def train(dataset, model, loss_fn, optim):
        ...

    options = {
      'model': Net(),
      'loss_fn': Loss(),
      'optimiser': SGD(),
      'dataset': Dataset(),
      'trainer': CustomTrainer()
    }

    job = DDPWrapper(platform=Platform.CPU, **options)
    job.start(epochs=20, ckpt_every=5, log_dir='./runs', ckpt_dir='./models')
    # job.resume(..., ckpt='./models/ckpt_7.pt')

Training on GPUs
----------------

Whether the model is to be trained on a single GPU or multiple, whether in a
cluster-based environment or cluster-free, PyTorch's ``DistributedDataParallel``
and the related classes are used.

1. **Cluster-less environment**. Training in a non-cluster environment requires
   spawning a thread for each GPU and assigning the threads to the GPUs.
   Spawning is done with PyTorch's ``multiprocessing``. The abstraction is
   provided by the custom class ``Parallelit``

   .. code-block:: python
      :linenos:
      :emphasize-lines: 5

      from ddpwrapper import DDPWrapper, Platform

      ...

      job = DDPWrapper(platform=Platform.CLGPU, ...)
      job.start(epochs=20, ckpt_every=5)

2. **SLURM-based cluster environment**. Training on a SLURM-based cluster
   requires high-level resources management. This is achieved with ``Submitit``

   .. code-block:: python
      :linenos:
      :emphasize-lines: 5

      from ddpwrapper import DDPWrapper, Platform

      ...

      job = DDPWrapper(platform=Platform.SLURM, ...)
      job.start(epochs=20, ckpt_every=5)