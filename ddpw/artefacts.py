from typing import final, Optional, Callable
from dataclasses import dataclass

from torch import nn, optim
from torch.utils import data

from .utils import Utils


OptimiserLoader = Callable[[nn.Module], optim.Optimizer]
r"""A callable that accepts a module and returns a PyTorch optimiser."""

@final
@dataclass
class ArtefactsConfig(object):
  r"""Configurations relating to the dataset and the model."""

  train_set: data.Dataset = None
  r"""The dataset to use for training. Default: ``None``."""

  test_set: data.Dataset = None
  r"""The dataset to use for evaluation. Default: ``None``."""

  validation_set: data.Dataset = None
  r"""The dataset to use for validation. Default: ``None``."""

  collate_fn: Optional[Callable] = None
  r"""Any callable function to be passed down to the dataloader. Default:
   ``None``."""

  validation_percentage: float = 0
  r"""The percentage of training dataset to be used for validation. If this
  property has a value of ``0``, it is assumed that no validation is required.
  This property is ignored for evaluation. Range: ``0`` to ``50``, inclusive.
  Default: ``0``. Range enforced by the :py:attr:`.needs_validation`
  property."""

  batch_size: int = 64
  r"""Batch size for training and testing. Default: ``64``."""

  model: nn.Module = None
  r"""An instance of the model to train. Default: ``None``."""

  model_has_batch_norm: bool = False
  r"""Specifies if the model to be trained has batch normalisation in it.
  Default: ``False``."""

  optimiser_loader: OptimiserLoader = None
  r"""Optimiser setup to be passed by the user. Default: ``None``."""

  optimiser: optim.Optimizer = None
  r"""
  .. admonition:: Definition not required
   :class: note

   This property need not be specified by the user and will be automatically
   updated by the wrapper right before training. This can be directly in the
   :py:meth:`.Job.train` method.

  The wrapper loads model parameters into the optimiser with the specified
  configs in :py:attr:`optimiser_loader` and updates this parameter. This could
  be accessed in :py:meth:`.Job.train` or :py:meth:`.Job.evaluate`.
  Default: ``None``.
  """

  @property
  def needs_validation(self):
    r"""
    This property tells if the current configuration requires validation or not.

    :returns bool: Whether validation is required or not.
    """

    return 0 < self.validation_percentage <= 50

  def print(self):
    r"""
    This method prints this object in a readable format.
    """

    Utils.print('Artefacts details:')
    Utils.print(' • Train dataset:                       ' +
                f'{len(self.train_set) if self.train_set is not None else 0}')
    Utils.print(' • Test dataset:                        ' +
                f'{len(self.test_set) if self.test_set is not None else 0}')
    if self.needs_validation:
      Utils.print(' • Validation percentage:               ' +
                  f'{self.validation_percentage}')
    Utils.print(f' • Batch size:                          {self.batch_size}')
