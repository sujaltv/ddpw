from abc import abstractclassmethod, ABC

import torch
import torch.distributed as dist
from torch.nn.modules.loss import _Loss as Loss


class EvalMetrics(object):
  r"""A template class for defining and computing accuracy metrics"""

  total_samples: torch.Tensor
  no_of_correct: torch.Tensor

  #: The number of processes to average over
  nprocs: int = 1

  def __init__(self):
    self.total_samples = torch.Tensor([0])
    self.no_of_correct = torch.Tensor([0])

  @property
  def accuracy(self):
    return (self.no_of_correct * 100) / (self.total_samples or 1) / self.nprocs

  def print(self):
    r"""A method to print accuracy metrics"""

    print(f'Total samples: {self.total_samples.item()}')
    print(f'Number of correct predictions: {self.no_of_correct.item()}')
    print('Accuracy: {:.2f}'.format(self.accuracy.item()))

  def reduce(self, dst):
    dist.reduce(self.total_samples.detach().clone(), dst=dst)
    dist.reduce(self.no_of_correct.detach().clone(), dst=dst)


class Trainer(ABC):
  r"""An abstract class requiring definitions of training and evaluating a
  model"""

  @abstractclassmethod
  def train(
    self,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Loss,
    optimiser: torch.optim.Optimizer
  ):
    r"""The train method defines the training procedure

    :param torch.nn.Module model: The model to be trained
    :param torch.utils.data.DataLoader dataloader: The dataload with training
      data
    :param Loss loss_fn: The loss function
    :param torch.optim.Optimizer optimiser: The optimiser
    """

    pass

  @abstractclassmethod
  def loss(
    self,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Loss
  ) -> torch.Tensor:
    r"""The loss method returns the model's loss without backpropagating

    :param torch.nn.Module model: The model to be trained
    :param torch.utils.data.DataLoader dataloader: The dataload with training
      data
    :param Loss loss_fn: The loss function
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
