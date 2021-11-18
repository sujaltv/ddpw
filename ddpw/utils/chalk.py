import sys
from enum import Enum


class chalk(object):
  r"""
  The :class:`chalk` class offers class methods for decorating console output

  **Example**

  .. code:: python

    # write underlined "Success" in bold green
    chalk.underline().bold().green().text('Success\n').write()
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
  def bold(cls):
    """Make the text bold"""
    return cls.anchor(cls.Colour.BOLD.value)

  @classmethod
  def underline(cls):
    """Underline the text"""
    return cls.anchor(cls.Colour.UNDERLINE.value)

  @classmethod
  def purple(cls):
    """Make the text purple"""
    return cls.anchor(cls.Colour.PURPLE.value)

  @classmethod
  def cyan(cls):
    """Make the text cyan"""
    return cls.anchor(cls.Colour.CYAN.value)

  @classmethod
  def dark_cyan(cls):
    """Make the text dark cyan"""
    return cls.anchor(cls.Colour.DARKCYAN.value)

  @classmethod
  def blue(cls):
    """Make the text blue"""
    return cls.anchor(cls.Colour.BLUE.value)

  @classmethod
  def green(cls):
    """Make the text green"""
    return cls.anchor(cls.Colour.GREEN.value)

  @classmethod
  def yellow(cls):
    """Make the text yellow"""
    return cls.anchor(cls.Colour.YELLOW.value)

  @classmethod
  def red(cls):
    """Make the text red"""
    return cls.anchor(cls.Colour.RED.value)

  @classmethod
  def text(cls, txt):
    """Text to be written to the standard output"""
    cls.items.append(txt + cls.Colour.END.value)
    return cls()

  @classmethod
  def write(cls):
    """Flush out the text with specified styles to the standard output"""
    sys.stdout.write(''.join(cls.items) + cls.Colour.END.value)
    cls.items = []
    return cls()
