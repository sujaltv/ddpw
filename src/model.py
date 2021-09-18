import torch


class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.a = torch.nn.Parameter(torch.tensor([0.]))
    self.b = torch.nn.Parameter(torch.tensor([0.]))
    self.c = torch.nn.Parameter(torch.tensor([0.]))

  def forward(self, x):
    return self.a * (x ** 2) + self.b * x + self.c
