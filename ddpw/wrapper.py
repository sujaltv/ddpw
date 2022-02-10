import os

import torch.multiprocessing as mp
from submitit import AutoExecutor, JobEnvironment

from .utils import Utils
from .trainer import Trainer
from .gpu_setup import init_process
from .artefacts import ArtefactsConfig
from .platform import Platform, PlatformConfig


class Wrapper(object):
  r"""
  This class provides encapsulation for training a model on a CPU, GPU, or a
  SLURM-based cluster of GPU nodes.

  :param PlatformConfig p: Platform-related configurations
  :param ArtefactsConfig a_config: Dataset- and model-related configurations
  """

  def __init__(self, p: PlatformConfig, a_config: ArtefactsConfig):
    Utils.verbose = p.verbose

    Utils.print('Initialising the DDP Wrapper.')
    self.p: PlatformConfig = p
    self.a_config: ArtefactsConfig = a_config

    if p.requires_ipc:
      try:
        mp.set_start_method(p.spawn_method)
      except RuntimeError as e:
        Utils.print(
          f'Warning: {e}. Skipping setting the start method for forks.')

  def __gpu(self, run: Trainer):
    r"""
    This method sets up the training setup for cluster-less GPUs.
    """

    if self.p.world_size == 1:
      Utils.print('[Device 0] Task starting on GPU.')
      init_process(0, 0, run, self.p, self.a_config)
      return

    Utils.print(f'Spawning {self.p.world_size} processes.')
    processes = []

    # create a process for each GPU in the world
    for global_rank in range(self.p.world_size):
      p = mp.Process(target=init_process,
                    args=(global_rank, global_rank, run, self.p, self.a_config))
      processes.append(p)
      p.start()

    for p in processes:
      p.join()

    Utils.print('All processes complete.')

  def __slurm(self, individual_gpu, console_logs_path: str = './logs'):
    r"""
    This method sets up the training setup for SLURM-based clusters of GPU
    nodes.

    :param Trainer run: Custom training/evaluation task
    :param str console_logs_path: Location to save console logs. Default:
        ``./logs``
    """

    Utils.print('Setting up the SLURM platform.')

    executor = AutoExecutor(folder=console_logs_path)
    executor.update_parameters(
      name=self.p.name,
      mem_gb=12*self.p.n_nodes,
      gpus_per_node=self.p.n_gpus,
      tasks_per_node=self.p.n_gpus,
      cpus_per_task=self.p.cpus_per_task,
      nodes=self.p.n_nodes,
      timeout_min=self.p.timeout_min,
      slurm_partition=self.p.partition
    )

    return executor.submit(individual_gpu)

  def start(self, run: Trainer):
    r"""
    This method begins the setup process for CPU/GPU/SLURM-based training and
    starts the task.

    :param Trainer run: Custom training/evaluation definitions
    """

    Utils.print(f'Selected platform: {self.p.platform.name}.')
    Utils.print('Starting process(es).')

    if self.p.platform == Platform.CPU:
      init_process(0, 0, run, self.p, self.a_config)

    elif self.p.platform == Platform.GPU:
      self.__gpu(run)
      Utils.print('GPU processes finished.')

    elif self.p.platform == Platform.SLURM:
      def individual_gpu():
        r"""
        This nested function is the starting point for each SLURM-based GPU.
        """

        self.p.master_addr = os.environ['HOSTNAME']
        job_env = JobEnvironment()

        Utils.print(f'Node {job_env.node}: Local rank: {job_env.local_rank};' +
                    f'Gloal rank: {job_env.global_rank}.')

        init_process(job_env.global_rank, job_env.local_rank, run, self.p,
                     self.a_config)

      job = self.__slurm(individual_gpu, run.t_config.console_logs_path)
      Utils.print(f'SLURM job "{self.p.name}" scheduled. Job ID: {job.job_id}.')

    Utils.print('All jobs finished.')
