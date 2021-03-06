import abc
from typing import final, Optional, Callable
from dataclasses import dataclass

from torch import nn, optim
from torch.utils import data
from torch.nn.modules.loss import _Loss as Loss


class OptimiserLoader(object):
  r"""
  This class allows usage of optimisers with desired configurations. Before it
  begins the training/evaluation process, it uses the optimiser specified in
  this class.

  Of course, this class could be extended to customise as desired: to use
  different optimiser-specific parameters such as the learning rate, weight
  decay, momentum, `etc.`, as well as defining a set of optimisers to choose
  from depending on the desired optimiser at run-time.
  """

  @abc.abstractmethod
  def __call__(self, model: Optional[nn.Module]) -> optim.Optimizer:
    r"""
    .. admonition:: Definition required
      :class: important

      This method needs to be explicitly defined by the user.

    This method receives the model whose parameters are to be loaded into the
    optimiser. Once configured as desired, it returns the optimiser.

    :param Optional[nn.Module] model: The model whose parameters are to be
      loaded into the optimiser.
    :returns optim.Optimizer: The desired optimiser with configurations set and
        model parameters loaded
    :raises NotImplementedError: Method not implemented.
    """

    raise NotImplementedError


@final
@dataclass
class ArtefactsConfig(object):
  r"""Configurations relating to the dataset and the model."""

  dataset_root: str = '/data/dataset'
  r"""Location of the dataset to be used to training or evaluation."""

  train_set: data.Dataset = None
  r"""The dataset to use for training. Default: ``None``."""

  test_set: data.Dataset = None
  r"""The dataset to use for evaluation. Default: ``None``."""

  validation_set: data.Dataset = None
  r"""The dataset to use for validation. Default: ``None``."""

  collate_fn: Optional[Callable] = None
  r"""Any callable function to be passed down to the dataloader. Default:
   ``None``."""

  validation_percentage: float = 20
  r"""The percentage of training dataset to be used for validation. If this
  property has a value of ``0``, it is assumed that no validation is required.
  This property is ignored for evaluation. Range: ``0`` to ``50``, inclusive.
  Default: ``20``. Range enforced by the :py:attr:`.needs_validation`
  property."""

  batch_size: int = 64
  r"""Batch size for training and testing. Default: ``64``."""

  model: nn.Module = None
  r"""An instance of the model to train. Default: ``None``."""

  model_has_batch_norm: bool = False
  r"""Specifies if the model to be trained has batch normalisation in it.
  Default: ``False``."""

  loss_fn: Loss = None
  r"""An instance of the :class:`Loss` module. Default: ``None``."""

  optimiser_config: OptimiserLoader = None
  r"""Optimiser setup to be passed by the user. Default: ``None``."""

  optimiser: optim.Optimizer = None
  r"""
  .. admonition:: Definition not required
   :class: note

   This property need not be specified by the user and will be automatically
   updated by the wrapper right before training. This can be directly in the
   :py:meth:`.Trainer.train` method.

  The wrapper loads model parameters into the optimiser with the specified
  configs in :py:attr:`optimiser_config` and updates this parameter. This could
  be accessed in :py:meth:`.Trainer.train` or :py:meth:`.Trainer.evaluate`.
  Default: ``None``.
  """

  @property
  def needs_validation(self):
    r"""
    This property tells if the current configuration requires validation or not.

    :returns bool: Whether validation is required or not.
    """

    return 0 < self.validation_percentage <= 50
