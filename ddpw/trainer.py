import os
import abc
from enum import Enum
from typing import final
from dataclasses import dataclass

import torch
from torch import optim
from torch.utils import data
from torch.nn.modules.loss import _Loss as Loss

from .utils import Utils
from .platform import PlatformConfig
from .artefacts import ArtefactsConfig


@final
class TrainingMode(Enum):
  r"""Modes of running the task."""

  TRAIN = 0
  r"""for training"""

  RESUME = 1
  r"""to resume training"""

  EVALUATE = 2
  r"""for testing"""


@final
@dataclass
class TrainingConfig(object):
  job_type: TrainingMode = TrainingMode.TRAIN
  r"""Type of job. Default: ``TrainingMode.TRAIN``"""

  start_at: int = 0
  r"""Epoch number from which to start training. Default: ``0``"""

  epochs: int = 25
  r"""Epoch number at which to stop training. Default: ``25``"""

  model_name_prefix: str = 'checkpoint'
  r"""Prefix for saving the models. Default: ``checkpoint``"""

  model_path: str = './models'
  r"""Location at which to store the models. Default: ``./models``"""

  console_logs_path: str = './logs/console_logs'
  r"""Location at which to store the console logs. Default:
   ``./logs/console_logs``"""

  training_logs_path: str = './logs/training_logs'
  r"""Location at which to store training logs.
  Default: ``./logs/training_logs``"""

  optimiser: optim.Optimizer = None
  r"""The optimiser to call. Default: ``None``"""

  loss_fn: Loss = None
  r"""An instance of the Loss function. Default: ``None``"""

  save_every: int = 5
  r"""Save a ckeckpoint every few epochs. If 0, ignored. Default: ``5``"""

  learning_rate: int = 0.1
  r"""The learning rate to be used for optimisation. Default: ``0.1``"""


@dataclass
class Trainer(object):
  r"""
  This is an abstract template class to be defined by the user. This class
  provides methods to simply define training and evaluation methods.
  """

  t_config: TrainingConfig = None
  r"""Training-related configurations. This property may be used to access
  training-specific aspects such as a optimisation stragegy (the optimiser) or
  the optimisation objective (energy/loss functions)"""

  artefacts: ArtefactsConfig = None
  r"""Model-related configuration. This property may be used to access models,
  datasets, etc. for training and evaluation"""

  p_config: PlatformConfig = None
  r"""Platform-related configuration."""

  @abc.abstractmethod
  def train(self, global_rank: int):
    """This method provides definition for the training procedure.

    :param int global_rank: Global rank of the current device

    :raises NotImplementedError: Training has not been implemented
    """

    raise NotImplementedError

  @abc.abstractmethod
  def evaluate(self, global_rank: int, dataset: data.DataLoader):
    """This method provides definition for the evaluation procedure.

    :param int global_rank: Global rank of the current device
    :param data.DataLoader dataset: The dataset to use for evaluation

    :raises NotImplementedError: Evaluation has not been implemented
    """

    raise NotImplementedError

  def save(self, epoch: int):
    r"""
    This method is called at every few epochs (as configured). Override this
    method to save more information. The state so saved is used by the
    :meth:`__restore` method.

    This method is called only for training and not evaluation.

    :param int epoch: The epoch number at which to save the training state
    """

    checkpoint = {
      'stopped_at': epoch,
      'model': self.artefacts.model.state_dict(),
      'optimiser': self.t_config.optimiser.state_dict()
    }
    torch.save(checkpoint, os.path.join(self.t_config.model_path,
                          f'{self.t_config.model_name_prefix}_{epoch}.pt'))

  def __restore(self, resume_at: int):
    r"""
    Restore training from a saved state.

    Args:
        resume_at (int): The epoch checkpoint whence to resume the training.
    """

    filename = f'{self.t_config.model_name_prefix}_{resume_at}.pt'
    file_path = os.path.join(self.t_config.model_path, filename)
    assert os.path.isfile(file_path)

    Utils.print(f'Loading model {file_path}')
    checkpoint = torch.load(file_path)
    self.t_config.save_every = checkpoint['stopped_at']
    self.artefacts.model.load_state_dict(checkpoint['model'])
    self.t_config.optimiser.load_state_dict(checkpoint['optimiser'])


  def __call__(self, global_rank: int):
    r"""
    When once the distributed data parallel setups are completed by the wrapper,
    this method is called. This method locally updates the dataset and model
    allotted for the current GPU in case of GPU- and SLURM-based platforms.

      "param int global_rank: The global rank of the device
      "param nn.Module model: The model to train
      "param data.DataLoader train_set: The training dataset
      "param data.DataLoader val_set: The validation dataset
      "param data.DataLoader test_set: The test dataset
    """

    Utils.print(
      f'Device {global_rank}: Copying model parameters to the optimiser.')
    self.t_config.optimiser = self.t_config.optimiser(
      self.artefacts.model.parameters(), lr=self.t_config.learning_rate)

    # if this task is resumption from or evaluation of a saved model, load it
    if self.t_config.job_type in [TrainingMode.RESUME, TrainingMode.EVALUATE]:
      Utils.print(f'Device {global_rank}: model load setup underway.')
      self.__restore(self.t_config.start_at)

    # whether to training (or resumption) or evaluate
    if self.t_config.job_type in [TrainingMode.TRAIN, TrainingMode.RESUME]:
      self.train(global_rank)
    else:
      self.evaluate(global_rank, self.artefacts.test_set)