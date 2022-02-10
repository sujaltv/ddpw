from torch.utils import data
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader, random_split

from ..utils import Utils
from ..artefacts import ArtefactsConfig
from ..platform import Platform, PlatformConfig


def sampler(dataset: data.Dataset, world_size: int, rank: int, batch_size: int,
            is_cpu: bool = False):
  r"""
  This function creates a sampler for the given process (i.e., rank) if
  necessary (i.e., if not CPU) and creates a dataloder from the sampler.

  :param data.Dataset dataset: The dataset from which to sample
  :param int world_size: The world size
  :param int rank: Current GPU
  :param int batch_size: Batch size
  :param bool is_cpu: Is the dataset for CPU. Default: `False`

  :returns data.Dataset: The dataset for the current process
  """

  smplr = None if is_cpu else DistributedSampler(dataset, world_size, rank=rank)
  result = DataLoader(dataset, batch_size, sampler=smplr, pin_memory=True)

  return result


def dataset_setup(rank: int, p: PlatformConfig, artefacts: ArtefactsConfig):
  r"""
  This function selects a portion of the dataset for the current GPU (i.e., the
  rank) and splits it into train and validation in case training.

  :param int rank: Rank of the current GPU
  :param PlatformConfig p: Platform configurations
  :param ArtefactsConfig artefacts: Job configurations

  :returns tuple: The training set, validation set, and test set
  """

  train_set: data.DataLoader = None
  val_set: data.DataLoader = None
  test_set: data.DataLoader = None

  is_cpu = p.platform == Platform.CPU
  batch_size = artefacts.batch_size

  # if the training dataset is provided
  if (train_set := artefacts.train_set) is not None:

    # if requested to set aside a portion of the training set for validation
    if artefacts.needs_validation:
      dataset_size = len(train_set)
      v_size = (dataset_size * artefacts.validation_percentage) // 100
      t_size = dataset_size - v_size
      Utils.print(
        f'[Device {rank}] Train size = {t_size}; validation size = {v_size}.')
      [train_set, val_set] = random_split(train_set, [t_size, v_size])
      val_set = sampler(val_set, p.world_size, rank, batch_size, is_cpu)

    train_set = sampler(train_set, p.world_size, rank, batch_size, is_cpu)

  # if the test dataset is provided
  if (test_set := artefacts.test_set) is not None:
    Utils.print(f'[Device {rank}] Test size  {len(test_set)}.')
    test_set = sampler(test_set, p.world_size, rank, batch_size, is_cpu)

  return train_set, val_set, test_set
