Example
#######

1. Configure the platform
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from ddpw.platform import Platform, PlatformConfig

    p_config = PlatformConfig(
        name='DDPW Example,
        verbose=True,
        spawn_method='spawn',
        partition='general,
        platform=Platform.SLURM,
        n_nodes=2,
        n_gpus=4,
        master_addr='localhost,
        master_port='1889,
        seed=1889,
        timeout_min=2880,
        cpus_per_task=1
    )

2. Configure the artefacts
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from ddpw.artefacts import ArtefactsConfig

    from src.dataset import CustomDataset
    from src.model import CustomModel

    a_config = ArtefactsConfig(
        train_set=CustomDataset(train=True),
        test_set=CustomDataset(train=False),
        validation_percentage=25,
        batch_size=64,
        model=CustomModel(),
        model_has_batch_norm=True
    )

4. Configure the job
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from ddpw.trainer import TrainingConfig, TrainingMode

    from src.loss import CustomLoss

    t_config = TrainingConfig(
        job_type=TrainingMode.TRAIN,
        start_at=0,
        epochs=50,
        model_name_prefix='custom-net,
        model_path='./models,
        console_logs_path='./logs/console_logs',
        training_logs_path='./logs/tensorboard_logs',
        save_every=5,
        learning_rate=0.01,
        loss_fn=CustomLoss(),
        optimiser=torch.optim.Adadelta
    )

5. Calling the job
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    w = Wrapper(p_config, a_config)
    task = CustomTrainer(t_config)

    w.start(task)
