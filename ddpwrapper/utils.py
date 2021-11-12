import torch


def add_click_options(options):
  def _add_click_options(func):
    for option in reversed(options):
      func = option(func)
    return func
  return _add_click_options


# Function copied from:
# https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068
def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)
