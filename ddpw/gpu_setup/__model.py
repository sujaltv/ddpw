import torch
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from ..utils import Utils


def model_setup(model: torch.nn.Module, global_rank: int, local_rank: int,
                has_batch_norm: bool, requires_ipc: bool = True):
  r"""
  This function moves the model to the specified rank, sets it as a
  DDP-trainable module, and synchronises batch normalisation layers, if any

  Args:
      model (torch.nn.Module): The model to be set up
      global_rank (int): The global rank of the GPU
      local_rank (int): The local rank of the GPU
      has_batch_norm (bool): if the model has batch normalisation layers in it
      requires_ipc (bool): if there are more than one GPU
  """

  Utils.print(f'Device {global_rank}. Moving model to {local_rank}')

  model = model.cuda(torch.device(local_rank))

  if requires_ipc:
    if has_batch_norm:
      model = SyncBatchNorm.convert_sync_batchnorm(model)

    model = DistributedDataParallel(model, device_ids=[local_rank],
                                    find_unused_parameters=True)

  return model
