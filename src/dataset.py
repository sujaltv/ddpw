import torch


class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    super(Dataset, self).__init__()

    get_data = lambda x: 1 * (x ** 2) + 2 * x - 2

    self.x = torch.rand(1000) * 6 - 4
    self.y = get_data(self.x)

  def __len__(self):
    return self.x.numel()

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
