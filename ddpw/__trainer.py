from abc import abstractclassmethod, ABC

import torch
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class EvalMetrics(object):
  r"""A template class for defining and computing accuracy metrics"""

  total_samples: int = 0
  no_of_correct: int = 0

  @property
  def accuracy(self):
    return (self.no_of_correct * 100) /(self.total_samples or 1)

  def print(self):
    r"""A method to print accuracy metrics"""

    print(f'Total samples: {self.total_samples}')
    print(f'Number of correct predictions: {self.no_of_correct}')
    print('Accuracy: {:.2f}'.format(self.accuracy))


class Trainer(ABC):
  r"""An abstract class requiring definitions of training and evaluating a
  model"""

  @abstractclassmethod
  def train(
    self,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Loss,
    optimiser: torch.optim.Optimizer,
    optim_step: LRScheduler = None
  ):
    r"""The train method defines the training procedure

    :param torch.nn.Module model: The model to be trained
    :param torch.utils.data.DataLoader dataloader: The dataload with training
      data
    :param Loss loss_fn: The loss function
    :param torch.optim.Optimizer optimiser: The optimiser
    :param LRScheduler,optional optim_step: Optimiser steps. Defaults to None.
    """

    pass

  @abstractclassmethod
  def evaluate(self, model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader) -> EvalMetrics:
    r"""Evaluation method to evaluate a trained (or under-training) model
    with the evaluation dataset in ``dataloader``

    :param torch.nn.Module) model: The model to be evaluated
    :param torch.utils.data.DataLoader dataloader: The dataset to use for
      evaluation
    """

    pass
