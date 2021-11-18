import os

from .__wrapper import DDPWrapper
from .__platform import Platform
from .__trainer import Trainer, EvalMetrics
from .__logger import Logger, LoggerType

from .utils.torchtils import optimizer_to


__version__ = os.path.dirname(os.path.abspath(__file__))
__version__ = os.path.join(__version__, "../version.txt")
if os.path.isfile(__version__):
  with open(__version__, "r", encoding="utf-8") as f:
    __version__ = f.read()
