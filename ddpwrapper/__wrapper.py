import os

import torch
import submitit
import torch.distributed as dist
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .__platform import Platform
from .__trainer import Trainer
from .__logger import Logger, LoggerType
from .__parallelit import AutoExecutor


class DDPWrapper(object):
  model_has_batchnorm: bool = False

  platform: Platform = Platform.CLGPU
  nprocs: int = 1

  world_size: int = 1
  seed: int = 123
  start_at: int = 0

  def __init__(self, platform: Platform, model: torch.nn.Module, dataset: torch.utils.data.Dataset,
               loss_fn: Loss, optimiser: torch.optim.Optimizer, trainer: Trainer,
               optimiser_step: LRScheduler = None, nprocs: int = 1):
    self.platform = platform
    assert nprocs > 0
    self.nprocs = nprocs

    self.model = model
    self.dataset = dataset
    self.loss_fn = loss_fn
    self.optimiser = optimiser
    self.trainer = trainer
    self.optimiser_step = optimiser_step

  def train(self, model, dataset, optimiser, loss_fn, optimiser_step, epochs,
              ckpt_every, pid=0, logger: Logger=None):
    # start training
    for e in range(self.start_at, epochs):
      loss = self.trainer.train(model, dataset, loss_fn, optimiser,
                                optimiser_step)

      # barrier and reduce all

      if logger is not None or True:
        if self.platform != Platform.CPU:
          dist.barrier()
          loss_reduced = loss.detach().clone()
          dist.reduce(loss_reduced, dst=0)
          loss_reduced = loss_reduced / self.nprocs
          if pid == 0:
            # print({'Loss': loss_reduced}, e)
            logger.log(LoggerType.Scalar, {'Loss': loss}, e)
        else:
          # print({'Loss': loss}, e)
          logger.log(LoggerType.Scalar, {'Loss': loss}, e)

      if (e > 0 and e % ckpt_every == 0) or (e == epochs - 1):
        if ((self.platform == Platform.CLGPU or \
          self.platform == Platform.SLURM) and pid == 0) or \
          self.platform == Platform.CPU:
          print(f'Saving at epoch {e}')
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

  def resume(self, epochs, ckpt_every, ckpt_dir, ckpt, logdir: str=None):
    file_path = os.path.join(ckpt_dir, f'ckpt_{ckpt}.pt')
    assert os.path.isfile(file_path)
    checkpoint = torch.load(file_path)
    self.start_at = checkpoint['epoch']
    self.model.load_state_dict(checkpoint['model'])
    self.optimiser.load_state_dict(checkpoint['optimiser'])
    self.start(epochs, ckpt_every, ckpt_dir, logdir)

  def start(self, epochs, ckpt_every, ckpt_dir, logdir: str=None):
    self.ckpt_dir = ckpt_dir
    options = {
        'model': self.model,
        'optimiser': self.optimiser,
        'loss_fn': self.loss_fn,
        'optimiser_step': self.optimiser_step,
        'epochs': epochs,
        'dataset': self.dataset,
        'ckpt_every': ckpt_every,
        'logdir': logdir
    }

    if self.platform == Platform.CPU:
      options['dataset'] = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=100,
                                                       pin_memory=False)
      device = torch.device('cpu')
      options['model'] = self.model.to(device)
      self.train(**options, logger=Logger(logdir) if logdir else logdir)
    elif self.platform == Platform.CLGPU:
      init_method = 'tcp://localhost:1640'

      executor = AutoExecutor()
      executor.update_parameters(init_method, self.train, self.nprocs)
      executor.submit(options)
    elif self.platform == Platform.SLURM:
      hostname = os.environ.get('HOSTNAME', None)
      gpus_per_node = int(os.environ.get('SLURM_GPUS_PER_NODE', 1))
      num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
      partition = os.environ.get('SLURM_JOB_PARTITION', 'general')

      init_method = f'tcp://{hostname}:1640'
      os.environ['MASTER_ADDR'] = hostname
      os.environ['MASTER_PORT'] = '1640'
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
      parallel_exec.update_parameters(init_method, self.train, world_size)

      def wrapper():
        job_env = submitit.JobEnvironment()
        options['local_rank'] = job_env.local_rank
        pid = int(job_env.global_rank)
        parallel_exec.dist_init(pid, options)

      job = executor.submit(wrapper)
      print('Job ID: ', job.job_id)
    else:
      raise TypeError("Invalid platform or no implementation")
