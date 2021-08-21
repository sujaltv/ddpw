from abc import ABC, abstractclassmethod
import os
import torch
import numpy as np

from enum import Enum

from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss as Loss


class Train(ABC):
  @abstractclassmethod
  def __call__(
    self,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Loss,
    optimiser: torch.optim.Optimizer,
    optim_step: LRScheduler = None):
    pass


class DeviceType(Enum):
  CPU = 0
  SingleGPU = 1
  MultiGPU = 2
  MultiNode = 3


class DDPTraining(object):
  model_has_batchnorm: bool = False

  device_type: DeviceType = DeviceType.MultiGPU

  world_size: int = 1
  seed: int = 123
  epochs: int = 10

  init_method: str = 'env://'
  MASTER_ADDR: str = 'localhost'
  MASTER_PORT: int = '9090'

  def __init__(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset,
               loss_fn: Loss, optimiser: Optimizer, train: Train,
               optimiser_step: LRScheduler = None):
    self.model = model
    self.dataset = dataset
    self.loss_fn = loss_fn
    self.optimiser = optimiser
    self.train = train
    self.optimiser_step = optimiser_step

    if self.init_method == 'env://':
      os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR',self.MASTER_ADDR)
      os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT',self.MASTER_PORT)


  def distribute(self, pid):
    # initialise distributed process group for multigpu or multinode
    if self.device_type in [DeviceType.MultiGPU, DeviceType.MultiNode]:
      dist.init_process_group(
        backend=dist.Backend.NCCL,
        init_method=self.init_method,
        world_size=self.world_size,
        rank=pid
      )

    # seed random generator methods to be the same across GPUs
    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    torch.cuda.manual_seed_all(self.seed)

    if self.device_type in [DeviceType.MultiGPU, DeviceType.MultiNode]:
      dist.barrier(device_ids=[pid])

    # get a distributed version of everything
    model = self.model
    dataset = self.dataset
    smplr = None

    if self.device_type != DeviceType.CPU:
      torch.cuda.set_device(pid)
      model = model.cuda(pid)
    elif self.device_type in [DeviceType.MultiGPU, DeviceType.MultiNode]:
      if self.model_has_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
      model = DistributedDataParallel(model, [pid])
      smplr = DistributedSampler(dataset,
                                   num_replicas=self.world_size,rank=pid)

    dataset = DataLoader(dataset, batch_size=100,
                         pin_memory=self.device_type != DeviceType.CPU,
                         sampler=smplr)

    optimiser = self.optimiser
    optimiser = optimiser(model.parameters(), lr=0.1)

    optimiser_step = self.optimiser_step
    if optimiser_step is not None:
      optimiser_step = optimiser_step(optimiser, 0.5, 0.9)

    # start training
    for e in range(self.epochs):
      loss = self.train(model, dataset, self.loss_fn, optimiser, optimiser_step)

      if pid == 0:
        print(f'Epoch {e}. Loss = {loss}')

    # if root node, get the parameters
    if pid == 0:
      # save model here
      pass

    # destroy distributed parallelism
    if self.device_type in [DeviceType.MultiGPU, DeviceType.MultiNode]:
      dist.destroy_process_group()

  def __call__(self, world_size: int = 1, device_type: DeviceType = None):
    assert world_size > 0
    self.world_size = world_size

    if device_type is not None:
      self.device_type = device_type

    if self.device_type in [DeviceType.MultiGPU, DeviceType.MultiNode]:
      mp.spawn(self.distribute, (), nprocs=self.world_size, join=True)
    else:
      self.world_size = 1
      self.distribute(0)
