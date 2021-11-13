from abc import abstractclassmethod, ABC

import torch
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class EvalMetrics(object):
  total_samples: int = 0
  no_of_correct: int = 0

  @property
  def accuracy(self):
    return (self.no_of_correct * 100) /(self.total_samples or 1)

  def print(self):
    print(f'Total samples: {self.total_samples}')
    print(f'Number of correct predictions: {self.no_of_correct}')
    print('Accuracy: {:.2f}'.format(self.accuracy))


class Trainer(ABC):

  @abstractclassmethod
  def train(
    self,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Loss,
    optimiser: torch.optim.Optimizer,
    optim_step: LRScheduler = None
  ): ...

  @abstractclassmethod
  def evaluate(self, model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader) -> EvalMetrics: ...
