import torch


def optimizer_to(optim, device):
  r"""
  This function offers a simple way to move all parameters of an optimiser or,
  effectively the optimiser itself, to the specified device. This method has
  been taken as is from a `solution
  <https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068>_`
  suggested on PyTorch's Discuss forum.

  Args:
      optim (torch.optim): [description]
      device ([type]): [description]
  """

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