from enum import Enum
from typing import Dict

from torch.utils.tensorboard.writer import SummaryWriter


class LoggerType(Enum):
  """This class defined the types of logging allowed"""

  #: Scalar values
  Scalar = 0

  #: Dictionary values
  Scalars = 1


class Logger(SummaryWriter):
  """Class inheriting from tensorboard's :class:`SummaryWriter` to extend custom
  functionality
  """

  def log(self, log_type: LoggerType, item: Dict[str, any], itr: int):
    r"""
    This method logs a specific item into the SummaryWriter logs.

    :param LoggerType log_type: Type of log, whether a scalar, a dictionary,
      etc.
    :param (Dict[str,any]) item: The item to be logged
    :param int itr: Iteration at which the log happened
    """

    if log_type == LoggerType.Scalar:
      for key, value in item.items():
        self.add_scalar(key, value, itr)
    elif log_type == LoggerType.Scalars:
      for key, value in item.items():
        self.add_scalars(key, value, itr)
