import os
from typing import Any, Callable

import torch.distributed as dist
import torch.multiprocessing as mp

from .platform import Device, Platform
from .utils import Utils
from . import functional as DF


def setup(global_rank: int, local_rank: int, platform: Platform,
          target: Callable[[int, int], Any]) :
    r"""
    This function is called at the beginning of the process in each device
    (CPU/GPU). Depending on the needs, this function establishes DDP communication
    protocols, seeds random number generators, and starts the given task.

    :param int global_rank: Global rank of the device.
    :param int local_rank: Local rank of the device.
    :param Platform platform: Platform-related configurations.
    :param Callable target: The function to call upon setup.
    """

    Utils.print(f'[Device {global_rank}] Initialising the process.')

    if platform.requires_ipc:
        os.environ['MASTER_ADDR'] = platform.master_addr
        os.environ['MASTER_PORT'] = str(platform.master_port)

        im = f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
        Utils.print(f'[Device {global_rank}] IPC at {im}.')

        dist.init_process_group(backend=platform.backend, init_method=im,
                            rank=global_rank, world_size=platform.world_size)

    # 1. Seed random number generators
    Utils.print(f'[Device {global_rank}] Seeding random number generators.')
    DF.seed_generators(platform.seed)

    # 2. Wait for all processes to synchronise and then start the task
    if platform.requires_ipc:
        msg = f'[Device {global_rank}] Training model on device {local_rank}.'
        Utils.print(msg)
        dist.barrier()

    # 3. Call the task
    Utils.print(f'[Device {global_rank}] All setup finished.')
    target(global_rank, local_rank)

    # 4. Cleanup
    if platform.requires_ipc: dist.destroy_process_group()
    Utils.print(f'[Device {global_rank}] Tasks on device complete.')
 

class Wrapper:
    r"""
    This class bootstraps the device setup for CPU, GPU, or a SLURM-based
    cluster of GPU nodes. Once platform-specific configurations are specified,
    the given task will be started.

    :param Platform platform: Platform-related configurations.
    """

    def __init__(self, platform: Platform):
        Utils.verbose = platform.verbose

        Utils.print('Initialising the DDP Wrapper.')
        self.platform = platform

        if platform.requires_ipc:
            try:
                mp.set_start_method(platform.spawn_method)
            except RuntimeError as e:
                Utils.print(
                  f'Warning: {e}. Skipping setting the start method for forks.')

    def __gpu(self, target: Callable[[int, int], Any]):
        r"""
        This method spins up a process for each GPU in the world. It assigns the
        task to be run on each process, `viz.`, distributing the datasets and
        models and commencing the task.
        
        :param Callable target: The function to call on each GPU upon setup.
        """
        from submitit import JobEnvironment

        if self.platform.world_size == 1:
            Utils.print('[Device 0] Task starting on GPU.')
            setup(0, 0, self.platform, target)
            return

        Utils.print(f'Spawning {self.platform.world_size} processes.')
        processes = []

        # create a process for each GPU in the world
        for global_rank in range(self.platform.world_size):
            p = mp.Process(target=setup, args=(global_rank, global_rank,
                                                      self.platform, target))
            processes.append(p)
            p.start()

        for p in processes: p.join()

        Utils.print('All processes complete.')

    def __slurm(self, target: Callable, console_logs: str):
        r"""
        Similar to :py:meth:`.__gpu` but for SLURM. An additional step includes
        spinning up a process for each node, done with ``submitit``.

        :param Callable target: The function to call on each GPU upon setup.
        :param str console_logs: Location to save SLURM console logs.
        """
        from submitit import AutoExecutor

        Utils.print('Setting up the SLURM platform.')

        executor = AutoExecutor(folder=console_logs)
        executor.update_parameters(
            name=self.platform.name,
            mem_gb=self.platform.ram,
            gpus_per_node=self.platform.n_gpus,
            tasks_per_node=self.platform.n_gpus,
            cpus_per_task=self.platform.n_cpus,
            nodes=self.platform.n_nodes,
            timeout_min=self.platform.timeout_min,
            slurm_partition=self.platform.partition
        )

        return executor.submit(target)

    def start(self, target: Callable[[int, int], Any]):
        r"""
        This method begins the setup process for CPU/GPU/SLURM-based jobs and
        commences the task.

        :param Callable[[int, int], Any] target: The task. A callable which
            accepts two integers: the global and the local rank of the device.
        """

        self.platform.print()

        Utils.print('Starting process(es).')

        def finished():
            if self.platform.upon_finish is not None:
                return self.platform.upon_finish()

        match self.platform.device:
            case Device.CPU | Device.MPS:
                setup(0, 0, self.platform, target)
                finished()
            case Device.GPU:
                self.__gpu(target)
                finished()
            case Device.SLURM:
                def individual_gpu():
                    r"""
                    This nested function is the starting point for each
                    SLURM-based GPU.
                    """
                    from submitit import JobEnvironment

                    self.platform.master_addr = os.environ['HOSTNAME']
                    job_env = JobEnvironment()

                    details = f"""
                    \r • Node: {job_env.node}.
                    \r • Global rank: {job_env.global_rank}.
                    \r • Local rank: {job_env.local_rank}.
                    """
                    Utils.print(details)

                    setup(job_env.global_rank, job_env.local_rank,
                                 self.platform, target)

                    if job_env.global_rank == 0: finished()

                job = self.__slurm(individual_gpu, self.platform.console_logs)

                details = f"""
                \r SLURM job "{self.platform.name}" scheduled;\
                \r job ID: {job.job_id}.
                \r See respective device logs for output on those devices.
                """

                print(details)

