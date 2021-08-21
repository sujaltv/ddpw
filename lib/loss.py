import torch


class SampleLoss(torch.nn.Module):
  def __init__(self):
    super(SampleLoss, self).__init__()

  def forward(self, x, y):
    return torch.abs(x - y).sum()
