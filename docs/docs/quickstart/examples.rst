Examples
########

Context
^^^^^^^^

Consider the following snippet for a quick understanding of the examples.

.. code-block:: python

    from torch.optim import Adadelta

    from .src import CustomNet, CustomDataset, CustomLoss, CustomTrainer

    model = CustomNet() # the model to be trained
    optimiser = torch.optim.Adadelta(model.parameters(), lr=.1) # an optimiser
    dataset = CustomDataset(root='./data', train=True) # the dataset

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

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.CPU, **options)
    job.start(epochs=50, ckpt_every=20, ckpt_dir='./models', batch_size=64,
      logdir='runs')

Training on a GPU
-----------------

.. code-block:: python

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.GPU, **options)
    job.start(...)

Training on mutiple cluster-less GPUs
-------------------------------------

.. code-block:: python

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

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.SLURM, **options)
    job.start(...)

Slurm-based environment variables are used to determine the number of clusters,
nodes, cluster partitions, *etc*. They may be passed as environment variables
through either the CLI or from a script. The following is an example of a shell
script.

.. code-block:: bash
    :caption: An example ``slurm.sh`` file run as ``sbatch slurm.sh``

    #!/bin/sh

    #SBATCH --output=ddp.out
    #SBATCH --error=ddp.err
    #SBATCH --nodes=2
    #SBATCH --gpus-per-node=4
    #SBATCH --mem=20GB
    #SBATCH --ntasks-per-node=4
    #SBATCH --partition=Extended
    #SBATCH --time=1-4

    source activate ddp
    python main.py train
    conda deactivate


Resuming
^^^^^^^^

A model whose training was stopped at an epoch may be continued to be trained
thereon.

This allows for a model trained on a CPU or a GPU to be continued to be trained
multiple GPUs or SLUM clusters or vice versa.

.. code-block:: python

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.GPU, **options)

    # start from a model saved at 50th epoch and train until 125th epoch
    # (train another 75 epochs)
    job.resume(epochs=125, ckpt=50, ckptdir='./models', ..)

Evaluation
^^^^^^^^^^

Custom :class:`evaulation metrics <ddpw.EvalMetrics>` may be defined. The
following example shows evaluation of a saved model.

.. code-block:: python

    from ddpw import DDPWrapper, Platform

    job = DDPWrapper(platform=Platform.GPU, **options)

    # evaluate the model saved at 125th epoch
    job.evaluate(ckpt=125, ckpt_dir='./models').print()
