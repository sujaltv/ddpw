from datetime import datetime

import torch
from torchvision import transforms as T
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler, StepLR

from ddpwrapper import DDPWrapper, Platform, Trainer, Logger, EvalMetrics

from src.model import Net
from src.dataset import Dataset
from src.loss import Loss


class CustomTrainer(Trainer):
  def train(self, model: Net,
              dataloader: torch.utils.data.DataLoader, loss_fn: Loss,
              optimiser: torch.optim.Optimizer, optim_step: LRScheduler = None):
    loss = torch.zeros(1)

    device = model.device if hasattr(model, 'device') else 'cpu'

    for _, (datapoints, labels) in enumerate(dataloader):
      optimiser.zero_grad() # reset the gradients
      loss = loss_fn(model(datapoints), labels.to(device))
      loss.backward()
      optimiser.step() # update the model parameters

    if optim_step is not None:
      optim_step.step()

    return loss

  def loss(self, model: Net, dataloader: torch.utils.data.DataLoader,
           loss_fn: Loss):
    loss = torch.zeros(1)

    device = model.device if hasattr(model, 'device') else 'cpu'

    with torch.no_grad():
      for _, (datapoints, labels) in enumerate(dataloader):
        loss += loss_fn(model(datapoints), labels.to(device))

    return loss

  def evaluate(self, model: Net, dataloader: torch.utils.data.DataLoader):
    evaluation = EvalMetrics()

    with torch.no_grad():
      model.eval()
      device = model.device if hasattr(model, 'device') else 'cpu'

      for _, (datapoints, labels) in enumerate(dataloader):
        preds = model(datapoints)
        evaluation.total_samples += labels.numel()
        evaluation.no_of_correct += (preds.argmax(1) == labels.to(device))\
          .int().sum().item()

    return evaluation


if __name__ == '__main__':
  model = Net()

  log_dir = f'runs/{datetime.now().strftime("%I:%M:%S%p_%a_%d_%b_%Y")}'

  transformations = T.Compose([
    T.ToTensor(),
    T.Resize((28,28)),
    T.Normalize((0.1307,), (0.3081,))
  ])

  optimiser = torch.optim.Adadelta(model.parameters(), lr=.01)
  options = {
    'model': model,
    'loss_fn': Loss(),
    'optimiser': optimiser,
    # 'optimiser_step': StepLR(optimiser, step_size=1, gamma=0.7),
    'dataset': Dataset(root='data', download=True, train=True, transform=transformations),
    'trainer': CustomTrainer(),
    'nprocs': 2
  }

  job = DDPWrapper(platform=Platform.GPU, **options)
  job.start(epochs=50, ckpt_every=5, ckpt_dir='./models/multigpu', batch_size=64, logdir=log_dir)
  # job.evaluate(ckpt_dir='models', ckpt=49).print()
  # job.resume(epochs=100, ckpt_every=25, ckpt_dir='./models', ckpt=50, logdir=log_dir)
  # print(list(job.model.parameters()))
