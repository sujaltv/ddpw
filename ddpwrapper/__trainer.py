from abc import abstractclassmethod, ABC

import torch
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

class Trainer(ABC):

  @abstractclassmethod
  def train(
    self,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Loss,
    optimiser: torch.optim.Optimizer,
    optim_step: LRScheduler = None,
    ):
    ...
