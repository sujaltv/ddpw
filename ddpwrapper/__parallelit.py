import typing as tp

import torch
import numpy as np
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .__logger import Logger
from .__trainer import Trainer


class AutoExecutor(object):
  nprocs = 1
  init_method: str = 'tcp://localhost:1640'
  seed = 1640

  def update_parameters(self, trainer: Trainer, init_method=None,
                        nprocs: int = 1):
    assert nprocs > 0
    self.nprocs = nprocs
    if init_method is not None: self.init_method = init_method
    self.trainer = trainer

  def dist_init(self, pid, args):
    dist.init_process_group(
      backend=dist.Backend.GLOO,
      init_method=self.init_method,
      world_size=self.nprocs,
      rank=pid
    )

    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    torch.cuda.manual_seed_all(self.seed)

    dist.barrier()

    gpu = args.get('local_rank', pid)

    torch.cuda.set_device(gpu)

    model = args['model']
    dataset = args['dataset']
    optimiser = args['optimiser']
    loss_fn = args['loss_fn']
    optimiser_step = args.get('optimiser_step', None)
    epochs = args['epochs']
    ckpt_every = args['ckpt_every']
    validate = args['validate']
    validation_dataset = args['validation_dataset']

    model.to(gpu)
    model = DistributedDataParallel(model, [pid])
    smplr = DistributedSampler(dataset, num_replicas=self.nprocs, rank=pid)
    dataloader = DataLoader(dataset, batch_size=100, pin_memory=True,
                         sampler=smplr)

    validation_dataloader = None
    if validate:
      smplr2 = DistributedSampler(validation_dataset, num_replicas=self.nprocs,
                                  rank=pid)
      validation_dataloader = DataLoader(validation_dataset,
                                         batch_size=len(validation_dataset),
                                         pin_memory=True, sampler=smplr2)

    optimiser.params = model.parameters()

    logger = Logger(args['logdir']) if pid == 0 else None

    self.trainer(model, dataloader, optimiser, loss_fn, optimiser_step, epochs,
                 ckpt_every, pid=pid, logger=logger, validate=validate,
                 validation_dataset=validation_dataloader)

    dist.barrier()
    dist.destroy_process_group()

  def cpu_init(self, **kwargs):
    kwargs['dataset'] = torch.utils.data.DataLoader(kwargs['dataset'],
                        batch_size=kwargs['batch_size'], pin_memory=False)
    if kwargs['validate']:
      kwargs['validation_dataset'] = torch.utils.data\
          .DataLoader(kwargs['validation_dataset'],
                batch_size=len(kwargs['validation_dataset']), pin_memory=False)
    device = torch.device('cpu')
    logdir = kwargs['logdir']
    logger = Logger(logdir) if logdir else logdir
    kwargs['model'] = kwargs['model'].to(device)
    del kwargs['logdir']
    del kwargs['batch_size']
    self.trainer(**kwargs, logger=logger)

  def submit(self, args: tp.Dict) -> None:
    mp.spawn(self.dist_init, (args,), nprocs=self.nprocs)
