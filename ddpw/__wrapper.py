import os

import torch
import submitit
import torch.distributed as dist
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .__platform import Platform
from .__trainer import Trainer, EvalMetrics
from .__logger import Logger, LoggerType
from .__parallelit import AutoExecutor

class DDPWrapper(object):
  r"""The :class:`DDPWrapper` class provides the highest level encapsulation for
  training models on different platforms.

  :param Platform platform: The platform to train on: whether to train on a
      CPU, a GPU, or on clusted nodes
  :param torch.nn.Module model: The model architecture to be trained
  :param torch.utils.data.Dataset dataset: The dataset to use for training
  :param Loss loss_fn: The loss function to be used
  :param torch.optim.Optimizer optimiser: The optimiser
  :param Trainer trainer: An instance of :class:`Trainer` with definitions of
      how to train and evaluate the model
  :param LRScheduler,optional optimiser_step: Optimiser step. Defaults to
      None.
  :param int,optional nprocs: Number of process to train on. Defaults to 1.
  :param bool,optional validate: Whether to validate training. Defaults to
      False.
  :param torch.utils.data.Dataset,optional validation_dataset: If validation
      is sought, the dataset to validare the training on. Defaults to None.
  """

  r"""Whether the model had batch normalisation layers in it"""
  model_has_batchnorm: bool = False

  r"""The platform to train on"""
  platform: Platform = Platform.GPU

  r"""The number of processes/GPUs"""
  nprocs: int = 1

  r"""The total number of GPUs across nodes to be used; used while on SLURM"""
  world_size: int = 1

  r"""The prot at which interprocess communication happens"""
  port: int = 1640

  r"""Seed to generare random values across GPUs"""
  seed: int = 1640

  r"""Starting epoch number"""
  start_at: int = 0

  def __init__(self, platform: Platform, model: torch.nn.Module,
               dataset: torch.utils.data.Dataset, loss_fn: Loss,
               optimiser: torch.optim.Optimizer, trainer: Trainer,
               optimiser_step: LRScheduler = None, nprocs: int = 1,
               validate = False, validation_dataset = None):
    self.platform = platform
    assert nprocs > 0
    self.nprocs = nprocs

    self.model = model
    self.dataset = dataset
    self.loss_fn = loss_fn
    self.optimiser = optimiser
    self.trainer = trainer
    self.optimiser_step = optimiser_step
    self.validate = validate
    self.validation_dataset = validation_dataset

  def train(self, model, dataset, optimiser, loss_fn, optimiser_step, epochs,
            ckpt_every, pid=0, logger: Logger=None, validate = False,
            validation_dataset = None):
    r"""This method starts training the model.

    :param int epochs: The number of epochs until which to train
    :param int ckpt_every: Number of epoch interval to save a model checkpoint
    :param int,optional pid: The process (or GPU) ID. Defaults to 0.
    :param Logger,optional logger: Logging. Defaults to None.
    """

    # start training
    validation_accuracy = None

    for e in range(self.start_at, epochs):
      loss = self.trainer.train(model, dataset, loss_fn, optimiser,
                                optimiser_step)

      if validate and logger is not None:
        assert validation_dataset is not None
        validation_accuracy = self.trainer.evaluate(model, validation_dataset)

      # barrier and reduce all
      def common_logger():
        logger.log(LoggerType.Scalar, {'Loss': loss}, e)
        if validation_accuracy is not None:
          logger.log(LoggerType.Scalar, {
            'ValAccuracy': validation_accuracy.accuracy
          }, e)

      if logger is not None or True:
        if self.platform != Platform.CPU:
          dist.barrier()
          loss_reduced = loss.detach().clone()
          dist.reduce(loss_reduced, dst=0)
          loss_reduced = loss_reduced / self.nprocs
          if pid == 0: common_logger()
        else: common_logger()

      if ckpt_every > 0:
        if (e > 0 and e % ckpt_every == 0) or (e == epochs - 1):
          if ((self.platform == Platform.GPU or \
            self.platform == Platform.SLURM) and pid == 0) or \
            self.platform == Platform.CPU:
            self.__save(e)

    if logger is not None:
      logger.close()

  def __save(self, epoch):
    assert self.ckpt_dir is not None
    checkpoint = {
      'epoch': epoch,
      'model': self.model.state_dict(),
      'optimiser': self.optimiser.state_dict()
    }
    torch.save(checkpoint, os.path.join(self.ckpt_dir, f'ckpt_{epoch}.pt'))

  def evaluate(self, ckpt_dir, ckpt) -> EvalMetrics:
    r"""This method evaluates a specific model (checkpoint)

    :param str ckpt_dir: The location in which the model is saved
    :param int ckpt: The checkpoint number

    :return: :class:`EvalMetrics` An instance of the evaluation metric
    """

    file_path = os.path.join(ckpt_dir, f'ckpt_{ckpt}.pt')
    assert os.path.isfile(file_path)
    checkpoint = torch.load(file_path)
    self.model.load_state_dict(checkpoint['model'])
    dataloader = torch.utils.data.DataLoader(self.dataset,
                                batch_size=20,
                                pin_memory=True, num_workers=4)
    return self.trainer.evaluate(self.model, dataloader)

  def resume(self, epochs, ckpt_every, ckpt_dir, ckpt, batch_size,
             logdir: str=None):
    r"""This method evaluates a specific model (checkpoint)

    :param int epochs: The number of epochs until which to train
    :param int ckpt_every: Number of epoch interval to save a model checkpoint
    :param str ckpt_dir: The location in which the model is saved
    :param int ckpt: The checkpoint number
    :param int batch_size: Training batch size.
    :param str log_dir: Log directory

    :return: :class:`EvalMetrics` An instance of the evaluation metric
    """

    file_path = os.path.join(ckpt_dir, f'ckpt_{ckpt}.pt')
    assert os.path.isfile(file_path)
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    self.start_at = checkpoint['epoch']
    self.model.load_state_dict(checkpoint['model'])
    self.optimiser.load_state_dict(checkpoint['optimiser'])
    self.start(epochs, ckpt_every, ckpt_dir, batch_size, logdir)

  def start(self, epochs, ckpt_every, ckpt_dir, batch_size=64, logdir: str=None):
    r"""Start or resume training

    :param int epochs: The number of epochs until which to train
    :param int ckpt_every: Number of epoch interval to save a model checkpoint
    :param str ckpt_dir: The location in which the model is saved
    :param int batch_size: Training batch size.
    :param str log_dir: Log directory

    :raises TypeError: If unimplemented platform is specified, raise an
      exception
    """

    self.ckpt_dir = ckpt_dir
    options = {
        'model': self.model,
        'optimiser': self.optimiser,
        'loss_fn': self.loss_fn,
        'optimiser_step': self.optimiser_step,
        'epochs': epochs,
        'dataset': self.dataset,
        'ckpt_every': ckpt_every,
        'logdir': logdir,
        'validate': self.validate,
        'batch_size': batch_size,
        'validation_dataset': self.validation_dataset
    }
    hostname = os.environ.get('HOSTNAME', 'localhost')
    init_method = f'tcp://{hostname}:{self.port}'

    if self.platform == Platform.CPU:
      executor = AutoExecutor()
      executor.update_parameters(self.train, init_method, self.nprocs)
      executor.cpu_init(**options)
    elif self.platform == Platform.GPU:
      executor = AutoExecutor()
      executor.update_parameters(self.train, init_method, self.nprocs)
      executor.submit(options)
    elif self.platform == Platform.SLURM:
      gpus_per_node = int(os.environ.get('SLURM_GPUS_PER_NODE', 1))
      num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
      partition = os.environ.get('SLURM_JOB_PARTITION', 'general')

      os.environ['MASTER_ADDR'] = hostname
      os.environ['MASTER_PORT'] = str(self.port)
      executor = submitit.AutoExecutor(logdir)
      executor.update_parameters(
        mem_gb=12*gpus_per_node,
        nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        tasks_per_node=gpus_per_node,
        slurm_partition=partition,
        cpus_per_task=2
      )

      world_size = gpus_per_node * num_nodes
      parallel_exec = AutoExecutor()
      parallel_exec.update_parameters(self.train, init_method, world_size)

      def wrapper():
        job_env = submitit.JobEnvironment()
        options['local_rank'] = job_env.local_rank
        pid = int(job_env.global_rank)
        parallel_exec.start_ddp(pid, options)

      job = executor.submit(wrapper)
      print('Job ID: ', job.job_id)
    else:
      raise TypeError("Invalid platform or no implementation")
