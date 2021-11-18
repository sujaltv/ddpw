import typing as tp
import os

import torch
import numpy as np
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .__logger import Logger
from .utils.chalk import chalk


class AutoExecutor(object):
  #: The number of processes (or GPUs)
  nprocs = 1

  #: Inter-process connexion address
  init_method: str = 'tcp://localhost:1640'

  #: Seed for generating random numbers across processes
  seed = 1640

  def update_parameters(self, trainer, init_method=None, nprocs: int = 1):
    assert nprocs > 0
    self.nprocs = nprocs
    if init_method is not None: self.init_method = init_method
    self.trainer = trainer

  def start_ddp(self, pid: int, options: tp.Tuple):
    mp.set_start_method('spawn')
    p = mp.Process(target=self.dist_init, args=(pid, options))
    self.init_method = \
      f'tcp://{os.environ["HOSTNAME"]}:{os.environ["MASTER_PORT"]}'
    p.start()
    p.join()

  def dist_init(self, pid, args):
    chalk.yellow().text('Initialising on: ').text(f'Device {pid}\n').write()
    chalk.dark_cyan().text(f'Communication via {self.init_method}\n').write()

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
    epochs = args['epochs']
    ckpt_every = args['ckpt_every']
    validate = args['validate']
    validation_dataset = args['validation_dataset']

    model.to(gpu)
    model = DistributedDataParallel(model, [pid])
    smplr = DistributedSampler(dataset, num_replicas=self.nprocs, rank=pid)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'],
                            pin_memory=True, sampler=smplr)
    validation_dataloader = None
    if validate:
      smplr2 = DistributedSampler(validation_dataset, num_replicas=self.nprocs,
                                  rank=pid)
      validation_dataloader = DataLoader(validation_dataset,
                                         batch_size=args['batch_size'],
                                         pin_memory=True, sampler=smplr2)

    optimiser.params = model.parameters()
    logger = Logger(args['logdir']) if pid == 0 else None

    chalk.bold().underline().yellow()\
      .text('Information on:').text(' ').text(f'Device {pid}:\n')
    chalk.yellow().text('No. of training batches on this device: ')\
      .text(f'{len(dataloader)}\n')

    if validate:
      chalk.yellow().text('No. of validation batches on this device: ')\
        .text(f'{len(validation_dataloader)}\n')
    chalk.text(f'Training commenced on device {pid}:\n').write()

    self.trainer(model, dataloader, optimiser, loss_fn, epochs, ckpt_every,
                 pid=pid, logger=logger, validate=validate,
                 validation_dataset=validation_dataloader)
    chalk.green().text(f'Training finished on device {pid}.\n').write()

    dist.barrier()
    dist.destroy_process_group()

  def cpu_init(self, **kwargs):
    kwargs['dataset'] = DataLoader(kwargs['dataset'],  pin_memory=False,
                                   batch_size=kwargs['batch_size'])
    if kwargs['validate']:
      kwargs['validation_dataset'] = DataLoader(kwargs['validation_dataset'],
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
