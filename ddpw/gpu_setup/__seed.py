import random

import torch
import numpy as np


def seed_generators(seed: int):
  r"""
  Seed all random number generators of various packages

  Args:
      seed (int): The seed to initialise
  """

  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)