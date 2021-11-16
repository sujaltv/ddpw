import sys
from enum import Enum

import torch


def optimizer_to(optim, device):
  r"""
  This function offers a simple way to move all parameters of an optimiser or,
  effectively the optimiser itself, to the specified device. This method has
  been taken as is from a `solution
  <https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068>_`
  suggested on PyTorch's Discuss forum.

  Args:
      optim (torch.optim): [description]
      device ([type]): [description]
  """

  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)


class click(object):
  """The :class:`click` class offers class methods for decorating console output
  """

  class Colour(Enum):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

  items = []

  @classmethod
  def anchor(cls, flag: Colour):
    cls.items.append(flag)
    return cls()

  @classmethod
  def bold(cls): return cls.anchor(cls.Colour.BOLD.value)

  @classmethod
  def underline(cls): return cls.anchor(cls.Colour.UNDERLINE.value)

  @classmethod
  def purple(cls): return cls.anchor(cls.Colour.PURPLE.value)

  @classmethod
  def cyan(cls): return cls.anchor(cls.Colour.CYAN.value)

  @classmethod
  def dark_cyan(cls): return cls.anchor(cls.Colour.DARKCYAN.value)

  @classmethod
  def blue(cls): return cls.anchor(cls.Colour.BLUE.value)

  @classmethod
  def green(cls): return cls.anchor(cls.Colour.GREEN.value)

  @classmethod
  def yellow(cls): return cls.anchor(cls.Colour.YELLOW.value)

  @classmethod
  def red(cls): return cls.anchor(cls.Colour.RED.value)

  @classmethod
  def text(cls, txt): return cls.anchor(txt + cls.Colour.END.value)

  @classmethod
  def write(cls):
    sys.stdout.write(''.join(cls.items) + cls.Colour.END.value)
    cls.items = []
    return cls()
