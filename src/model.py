import torch
from torch.nn import functional as F


class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = torch.nn.Conv2d(1, 32, (3,3))
    self.conv2 = torch.nn.Conv2d(32, 64, (3,3))

    self.drop1 = torch.nn.Dropout2d(0.25)
    self.drop2 = torch.nn.Dropout2d(0.5)

    self.fc1 = torch.nn.Linear(9216, 128)
    self.fc2 = torch.nn.Linear(128, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.drop1(F.max_pool2d(x, 2))

    x = x.flatten(1)
    x = F.relu(self.fc1(x))
    x = self.drop2(x)
    x = self.fc2(x)

    return F.log_softmax(x, 1)
