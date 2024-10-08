from random import seed as rand_seed
from typing import Optional

from numpy.random import seed as np_seed
from torch import Tensor
from torch import device as t_device
from torch import distributed as dist
from torch.cuda import current_device, is_available, manual_seed_all
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, Module, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.random import manual_seed
from torch.utils.data import Dataset, DistributedSampler

from .platform import Device, Platform


def seed_generators(seed: int):
    r"""
    This function seeds [pseudo]random number generators from various packages.

    :param int seed: The seed.
    """

    rand_seed(seed)
    np_seed(seed)
    manual_seed(seed)
    manual_seed_all(seed)


def average_params_grads(module: Module, params: bool = False, grads: bool = True):
    r"""
    Given a module, this function averages the parameters of the given model
    and/or their gradients across all the GPUs (copied over from a `PyTorch blog
    <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`_ and further
    modified here).

    :param torch.nn.Module module: The module whose parameters/gradients are to
        be averaged.
    :param bool params: Whether to average the parameters or not. Default:
        ``False``.
    :param bool grads: Whether to average the gradients or not. Default:
        ``True``.
    """

    if not (params and grads):
        return

    world_size = float(dist.get_world_size())

    for p in module.parameters():
        if p.grad is not None and grads:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size
        if p is not None and params:
            dist.all_reduce(p, op=dist.ReduceOp.SUM)
            p /= world_size


def optimiser_to(optimiser: Optimizer, device: t_device):
    r"""
    This function offers a simple way to move all the parameters optimised by an
    optimiser to the specified device. This function has been taken as is from a
    `solution on PyTorch Discuss
    <https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3>`_.

    :param torch.optim.Optimizer optimiser: The optimiser to move to a device.
    :param torch.device device: The device to which to move the optimiser.
    """

    for param in optimiser.state.values():
        # if any global tensors in the state dict
        if isinstance(param, Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def has_batch_norm(module: Module) -> bool:
    r"""
    This function checks if a module has batch normalisation layer(s) in it.

    :param nn.Module module: The module to be checked for containing any batch
        normalisation layers.

    :returns bool: Whether or not the module has batch normalisation layer(s) in
        it.
    """

    if isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
        return True

    for child in module.children():
        if has_batch_norm(child):
            return True

    return False


def to(
    module: Module,
    local_rank: int,
    sync_modules: bool = True,
    device: Device = Device.GPU,
) -> Module:
    r"""
    A quick and minimal function that works on all devices to move the given
    module to the specified device: CPU, GPU, or MPS. If the platform is set up
    in an IPC fashion, this function optionally moves the module in a
    distributed data parallel-fashion and synchronises batch normalisation
    layers, if any.

    :param torch.nn.Module module: The module to be moved.
    :param int local_rank: The local rank of the device.
    :param bool sync_modules: Whether to synchronise the modules across devices
        or not. If yes, the module becomes ``DistributedDataParallel``. Default:
        ``True``.
    :param Device device: The type of device to which to move the module. This
        argument is useful because once ``device`` has been globally set, this
        function can be called regardless of the current device in which the
        items are and thus helps avoid additional checks. Default:
        ``Device.GPU``.

    :returns torch.nn.Module: The module moved to the appropriate device.
    """

    match device:
        case Device.CPU | Device.MPS:
            module = module.to(device.value)
        case _:
            module = module.cuda(t_device(local_rank))

    if sync_modules and device not in [Device.CPU, Device.MPS]:
        if has_batch_norm(module):
            module = SyncBatchNorm.convert_sync_batchnorm(module)

        if dist.is_initialized():
            module = DistributedDataParallel(
                module, device_ids=[local_rank], find_unused_parameters=False
            )

    return module


def get_dataset_sampler(
    dataset: Dataset, global_rank: int, platform: Platform
) -> Optional[DistributedSampler]:
    r"""
    This function selects a portion of the original dataset shared by other
    devices. If the device is CPU or MPS, no sharing is necessary.

    :param data.Dataset dataset: The dataset from which to sample for the
        current device.
    :param int global_rank: The global rank of the device.
    :param Platform platform: Platform-related configurations.

    :returns DistributedSampler: Dataset sampler for the given dataset and
        world size.
    """

    sampler = None

    if platform.device not in [Device.CPU, Device.MPS]:
        sampler = DistributedSampler(
            dataset, num_replicas=platform.world_size, rank=global_rank
        )

    return sampler


def device(module: Module) -> t_device:
    r"""
    Given a module, this function returns the device on which it currently
    resides. If the module has no parameters, the current device is returned by
    default.

    :param nn.Module module: The module whose device is sought.

    :returns torch.device: The device of the module.
    """

    from torch.backends.mps import is_available as is_mps_available

    p = module.parameters()
    try:
        device = next(p).device
    except StopIteration:
        if is_available():
            device = t_device(f"cuda:{current_device()}")
        elif is_mps_available():
            device = t_device("mps")
        else:
            device = t_device("cpu")

    return device
