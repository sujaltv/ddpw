from datetime import datetime

import torch
from torchvision import transforms as T

from src.model import Net
from src.dataset import Dataset
from src.loss import Loss
from src.trainer import CustomTrainer
from ddpwrapper import DDPWrapper, Platform


if __name__ == '__main__':
  model = Net()

  log_dir = f'runs/{datetime.now().strftime("%I:%M:%S%p_%a_%d_%b_%Y")}'

  transformations = T.Compose([
    T.ToTensor(),
    T.Resize((28,28)),
    T.Normalize((0.1307,), (0.3081,))
  ])

  pre_split = Dataset(root='data', download=True, train=True, transform=transformations)
  train_set, val_set = torch.utils.data.random_split(pre_split, [50000, 10000])

  optimiser = torch.optim.Adadelta(model.parameters(), lr=.034)
  options = {
    'model': model,
    'loss_fn': Loss(),
    'optimiser': optimiser,
    # 'optimiser_step': StepLR(optimiser, step_size=1, gamma=0.7),
    'dataset': train_set,
    'trainer': CustomTrainer(),
    # 'validate': True,
    # 'validation_dataset': val_set,
    'nprocs': 1
  }

  job = DDPWrapper(platform=Platform.CPU, **options)
  job.start(epochs=50, ckpt_every=5, ckpt_dir='./models/mulgpu', batch_size=64, logdir=log_dir)
  # job.evaluate(ckpt_dir='models', ckpt=49).print()
  # job.resume(epochs=100, ckpt_every=25, ckpt_dir='./models', ckpt=50, logdir=log_dir)
