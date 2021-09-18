from datetime import datetime

import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ddpwrapper import DDPWrapper, Platform, Trainer, Logger

from src.model import Net
from src.dataset import Dataset
from src.loss import Loss


class CustomTrainer(Trainer):
  def train(self, model: Net,
              dataloader: torch.utils.data.DataLoader, loss_fn: Loss,
              optimiser: torch.optim.Optimizer, optim_step: LRScheduler = None):
    loss = torch.zeros(1)

    for _, (datapoints, labels) in enumerate(dataloader):
      optimiser.zero_grad() # reset the gradients
      loss = loss_fn(model(datapoints), labels.to(model.device if hasattr(model, 'device') else 'cpu'))
      loss.backward()
      optimiser.step() # update the model parameters

    if optim_step is not None:
      optim_step.step()

    return loss


if __name__ == '__main__':
  model = Net()

  log_dir = f'runs/{datetime.now().strftime("%I:%M:%S%p_%a_%d_%b_%Y")}'

  options = {
    'model': model,
    'loss_fn': Loss(),
    'optimiser': torch.optim.AdamW(model.parameters(), lr=0.1),
    'dataset': Dataset(),
    'trainer': CustomTrainer(),
    'nprocs': 3
  }

  job = DDPWrapper(platform=Platform.CLGPU, **options)
  job.start(epochs=100, ckpt_every=25, ckpt_dir='./models', logdir=log_dir)
  # job.resume(epochs=100, ckpt_every=25, ckpt_dir='./models', ckpt=50, logdir=log_dir)
  # print(list(job.model.parameters()))
