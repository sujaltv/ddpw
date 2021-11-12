Examples
########

Context
^^^^^^^^

.. code-block:: python
    :linenos:

    from torch.optim import Adadelta

    from .src import CustomNet, CustomDataset, CustomLoss, CustomTrainer

    model = CustomNet()
    optimiser = torch.optim.Adadelta(model.parameters(), lr=.1)
    dataset = Dataset(root='data', download=True, train=True)

    options = {
      'model': model,
      'loss_fn': CustomLoss(),
      'optimiser': optimiser,
      'dataset': dataset,
      'trainer': CustomTrainer(),
      'nprocs': 1 # number of processes/CPUs/GPUs
    }

Training
^^^^^^^^

Training on a single CPU
------------------------

.. code-block:: python
    :linenos:

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.CPU, **options)
    job.start(epochs=50, ckpt_every=20, ckpt_dir='./models', batch_size=64,
      logdir='runs')

Training on a GPU
-----------------

.. code-block:: python
    :linenos:

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.GPU, **options)
    job.start(...)

Training on mutiple cluster-less GPUs
-------------------------------------

.. code-block:: python
    :linenos:

    from ddpw import DDPWrapper, Platform

    options = {
      ...
      'nprocs': 4 # train on 4 GPUs
    }

    job = DDPWrapper(platform=Platform.GPU, **options)
    job.start(...)

Training on SLURM-based GPU clusters
------------------------------------

.. code-block:: python
    :linenos:

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.SLURM, **options)
    job.start(...)

Slurm based environment variables are used to determin the number of clusters,
nodes, cluster partitions, *etc*. They may be passed as environment variables
through either the CLI or from a script.

Resuming
^^^^^^^^

A model whose training was stopped at an epoch may be continued to be trained
thereon.

This allows for a model trained on a CPU or a GPU to be continued to be trained
multiple GPUs or SLUM clusters or vice versa.

.. code-block:: python
    :linenos:

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.GPU, **options)

    # start from a model saved at 50th epoch and train until 125th epoch
    # (train another 75 epochs)
    job.resume(epochs=125, ckpt=50, ckptdir='./models', ..)

Evaluation
^^^^^^^^^^

Custom evaluation metrics may be defined. The following example shows evaluation
of a saved model.

.. code-block:: python
    :linenos:

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.GPU, **options)

    # evaluate the model saved at 125th epoch
    job.evaluate(ckpt=125, ckpt_dir='./models').print()
