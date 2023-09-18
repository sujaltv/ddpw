from enum import Enum
from random import randint
from typing import final, Optional, Callable, Union
from dataclasses import dataclass

from torch import distributed as dist


@final
class Device(Enum):
    r"""The device on which to train or evaluate."""

    CPU = 'cpu'
    r"""The device to run on is a CPU."""

    GPU = 'gpu'
    r"""The device to run on is one or more GPUs."""

    SLURM = 'slurm'
    r"""The device to run on is a cluster of GPU nodes managed by SLURM."""

    MPS = 'mps'
    r"""The device to run on is an Apple SoC."""

    @staticmethod
    def from_str(device: str) -> 'Device':
        r"""
        This method returns a :py:class:`Device` object given a valid device
        string.

        :param str device: The name of the device. Supported values: ``cpu``,
            ``gpu``, ``slurm``, and ``mps`` (case insensitive).

        :returns Device: :py:class:`Device` corresponds to the device string.
        
        :raises ValueError: Raises an error if the string is invalid.
        """

        match device.lower():
            case 'cpu': return Device.CPU
            case 'gpu': return Device.GPU
            case 'slurm': return Device.SLURM
            case 'mps': return Device.MPS
            case _: raise ValueError("Invalid device string specified.")


@final
@dataclass
class Platform:
    r"""
    Platform-related configurations such as the device, environment,
    communication IP address and port, world size, `etc.`
    """

    name: str = 'ddpw'
    r"""Name of the platform job. Used by SLURM. Default: ``ddpw``."""

    device: Union[Device, str] = Device.GPU
    r"""The type of device. Default: ``Device.GPU``."""

    partition: str = 'general'
    r"""Name of the partition. Used by SLURM. Default: ``general``."""

    n_nodes: int = 1
    r"""The total number of nodes (used by SLURM).  Default: ``1``."""

    n_gpus: int = 1
    r"""The total number of GPUs (in each node). Default: ``1``."""

    n_cpus: int = 1
    r"""The total number of CPUs (per task/thread). Default: ``1``."""

    ram: int = 32
    r"""RAM per CPU in GB (used by SLURM). Default: ``32``."""

    spawn_method: Optional[str] = 'fork'
    r"""This property corresponds to that passed to
    :meth:`mp.set_start_method`. Default: ``fork``."""

    master_addr: str = 'localhost'
    r"""IPC address. Default: ``localhost``."""

    master_port: Optional[str] = str(randint(1024, 49151))
    r"""The port at which IPC happens. Default: a random port between 1024 and
    49151."""

    backend: Optional[dist.Backend] = dist.Backend.GLOO if hasattr(dist, 'Backend') else None
    r"""The PyTorch-supported backend to use for distributed data parallel.
    Default: ``torch.distributed.Backend.GLOO``."""

    seed: int = 1889
    r"""Seed with which to initialise the various [pseudo]random number
    generators. Default: ``1889``."""

    timeout_min: int = 2880
    r"""Minimum timeout (in minutes) for jobs (used by SLURM). Default:
    ``2880`` (two days)."""

    console_logs: str = './logs'
    r"""Location of console logs (used by SLURM). Default: ``./logs``"""

    verbose: Optional[bool] = True
    r"""Whether to run the wrapper in a verbose mode or not. Default:
    ``True``."""

    upon_finish: Optional[Callable] = None
    r"""Any cleanup tasks to be done upon completion. Default: ``None``."""

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = Device.from_str(self.device)

    @property
    def world_size(self):
        r"""World size. This is the total number of GPUs across all nodes.
        Default: ``1``."""

        return (self.n_nodes if self.device == Device.SLURM else 1) \
                * self.n_gpus

    @property
    def requires_ipc(self):
        r"""Needs communication. This property says whether or not the setup
        requires IPC. IPC is not required for a single device."""

        return self.device not in [Device.CPU, Device.MPS] \
                and self.world_size > 1

    def print(self):
        r"""
        This method prints this object in a human readable format.
        """

        details = f"""
        \r Platform details:

        \r • Name:\t\t\t\t{self.name}
        \r • Device:\t\t\t\t{self.device.value.upper()}
        \r • CPUs (per thread):\t\t\t{self.n_cpus}
        \r • RAM (per CPU):\t\t\t{self.ram}GB
        \r • GPUs (per node):\t\t\t{self.n_gpus}
        \r • PyTorch backend:\t\t\t{self.backend}
        \r • Seed (random number generators):\t{self.seed}
        """

        if self.device == Device.SLURM or True:
            details += f"""\
            \r • Nodes:\t\t\t\t{self.n_nodes}
            \r • SLURM partition:\t\t\t{self.partition}
            \r • SLURM timeout:\t\t\t{self.timeout_min} minutes
            """

        if not self.requires_ipc:
            details += f"""\
            \r • Spawn method:\t\t\t{self.spawn_method}
            \r • Master IP address:\t\t\t{self.master_addr}
            \r • Master port:\t\t\t\t{self.master_port}
            """

        details += f"""\
        \r • World size:\t\t\t\t{self.world_size}
        """

        print(details)

