import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler, StepLR

from src.loss import Loss
from src.model import Net
from ddpwrapper import Trainer, EvalMetrics


class CustomTrainer(Trainer):
  def train(self, model: Net,
              dataloader: torch.utils.data.DataLoader, loss_fn: Loss,
              optimiser: torch.optim.Optimizer, optim_step: LRScheduler = None):
    loss = torch.zeros(1)

    device = torch.device('cpu')
    if hasattr(model, 'device'): device = torch.device(model.device)

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
