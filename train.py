import os
import torch

from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.lr_scheduler import StepLR

from lib.dataset import SampleDataset
from lib.model import SampleModel
from lib.loss import SampleLoss
from lib.trainer import Train, DDPTraining, DeviceType


class SampleTrain(Train):
  def __call__(self, model: SampleModel,
              dataloader: torch.utils.data.DataLoader, loss_fn: SampleLoss,
              optimiser: torch.optim.Optimizer, optim_step: LRScheduler = None):
    if not hasattr(model, 'device'):
      if next(model.parameters()).is_cuda: device = torch.device('cuda')
      else: device = torch.device('cpu')
    else:
      device = model.device
    loss = torch.zeros(1)

    for _, (datapoints, labels) in enumerate(dataloader):
      optimiser.zero_grad() # reset the gradients
      loss = loss_fn(model(datapoints.to(device)), labels.to(device))
      loss.backward()
      optimiser.step() # update the model parameters

    if optim_step is not None:
      optim_step.step()

    return loss.item()



if __name__ == '__main__':
  visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '5')

  nprocs = len(visible_devices.split(','))

  model = SampleModel()
  dataset = SampleDataset()
  optimiser = torch.optim.AdamW
  loss_fn = SampleLoss()
  optimiser_step = StepLR
  train  = SampleTrain()
  epochs = 10

  distributed_training = DDPTraining(model, dataset, loss_fn, optimiser, train,
                                     optimiser_step)
  distributed_training(nprocs, DeviceType.MultiGPU)
