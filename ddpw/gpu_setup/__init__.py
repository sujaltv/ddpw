import os

import torch.distributed as dist

from ..utils import Utils
from ..trainer import Trainer
from ..artefacts import ArtefactsConfig
from ..platform import Platform, PlatformConfig

from .__model import model_setup as __model_setup
from .__dataset import dataset_setup as __dataset_setup
from .__seed import seed_generators as __seed_generators


def init_process(global_rank: int, local_rank: int, run: Trainer,
                 p: PlatformConfig, artefacts: ArtefactsConfig):
  r"""
  This function is called at the beginning of the process in each GPU. This
  function establishes DDP communication protocols, selects a portion of the
  data for the current GPU, moves the model to the device, and starts the given
  task.

  Args:
      global_rank (int): The global rank of the GPU
      local_rank (int): The local rank of the GPU
      run (Job): The task to run when once the setup is complete
      p (PlatformConfig): Platform-related configurations
      artefacts (ArtefactsConfig): Model-related configurations
  """

  Utils.print(f'Device {global_rank}. Initialising the process.')

  if p.requires_ipc:
      os.environ['MASTER_ADDR'] = p.master_addr
      os.environ['MASTER_PORT'] = f'{p.master_port}'

      Utils.print(f'Device {global_rank}. ' +
        f'IPC at tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}')

      dist.init_process_group(backend=p.backend, rank=global_rank,
                              world_size=p.world_size)

  # 0. Seed random number generators
  Utils.print(f'Device {global_rank}. Seeding random number generators.')
  __seed_generators(p.seed)

  # 1. organise the dataset into splits
  Utils.print(f'Device {global_rank}. Selecting portion of the dataset')
  artefacts.train_set, artefacts.validation_set, artefacts.test_set = \
    __dataset_setup(global_rank, p, artefacts)

  # 2. Set up the model on the current device
  if p.platform != Platform.CPU:
    Utils.print(f'Device {global_rank}. Copying the model to GPU {local_rank}')
    artefacts.model = __model_setup(artefacts.model, global_rank, local_rank,
                          artefacts.model_has_batch_norm, p.requires_ipc)

  # 3. Wait for all processes to synchronise and then start the task
  Utils.print(f'Device {global_rank}. Training model on device {global_rank}')
  if p.requires_ipc:
    dist.barrier()


  Utils.print(f'Device {global_rank}: all setup finished.')
  run.p_config = p
  run.artefacts = artefacts
  run(global_rank)

  if p.requires_ipc:
    dist.destroy_process_group()

  Utils.print('Tasks on device complete')
