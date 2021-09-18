import torch


class Loss(torch.nn.Module):
  def __init__(self):
    super(Loss, self).__init__()

  def forward(self, x, y):
    return torch.abs(x - y).sum()
