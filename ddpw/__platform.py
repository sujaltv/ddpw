from enum import Enum


class Platform(Enum):
  r"""The platform on which to run the training"""

  #: Indicates the platform to run on is a CPU
  CPU = 0

  #: The platform to run on is GPU
  GPU = 1

  #: The platform to run on is a SLURM cluster
  SLURM = 2
