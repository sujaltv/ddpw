from enum import Enum
from typing import Dict

from torch.utils.tensorboard.writer import SummaryWriter


class LoggerType(Enum):
  Scalar = 0
  Scalars = 1


class Logger(SummaryWriter):
  def log(self, log_type: LoggerType, item: Dict[str, any], itr: int):
    if log_type == LoggerType.Scalar:
      for key, value in item.items():
        self.add_scalar(key, value, itr)
    elif log_type == LoggerType.Scalars:
      for key, value in item.items():
        self.add_scalars(key, value, itr)
