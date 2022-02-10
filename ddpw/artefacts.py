from typing import final
from dataclasses import dataclass

from torch import nn
from torch.utils import data


@final
@dataclass
class ArtefactsConfig(object):
  r"""Configurations relating to the dataset and the model."""

  dataset_root: str = '/data/dataset'
  r"""Location of the dataset to be used to training or evaluation"""

  train_set: data.DataLoader = None
  r"""The dataset to use for training. Default: ``None``"""

  test_set: data.DataLoader = None
  r"""The dataset to use for evaluation. Default: ``None``"""

  validation_set: data.DataLoader = None
  r"""The dataset to use for validation. Default: ``None``"""

  validation_percentage: float = 20
  r"""The percentage of training dataset to be used for validation. If this
  property has a value of ``0``, it is assumed that no validation is required.
  This property is ignored for evaluation. Range: ``0`` to ``50``, inclusive.
  Default: ``20``. Range enforced by the :py:attr:`.needs_validation`
  property"""

  batch_size: int = 64
  r"""Batch size for training and testing. Default: ``64``"""

  model: nn.Module = None
  r"""An instance of the model to train. Default: ``None``"""

  model_has_batch_norm: bool = False
  r"""Specifies if the model to be trained has batch normalisation in it.
  Default: ``False``"""

  @property
  def needs_validation(self):
    r"""
    This property tells if the current configuration requires validation or not.

    :returns bool: Whether validation is required or not
    """

    return 0 < self.validation_percentage <= 50
