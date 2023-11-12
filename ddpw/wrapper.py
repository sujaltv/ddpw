import os
from os.path import isabs
from typing import Any, Callable, Optional, Tuple

import torch.distributed as dist
import torch.multiprocessing as mp

from .platform import Device, Platform
from .utils import Utils
from . import functional as DF


def setup(node: int, global_rank: int, local_rank: int, platform: Platform,
          target: Callable[[int, int, Optional[Tuple]], Any],
          args: Optional[Tuple]) :
    r"""
    This function is called at the beginning of the process in each device
    (CPU/GPU). Depending on the needs, this function establishes DDP
    communication protocols, seeds random number generators, and invokes the
    given task.

    :param int node: Node number.
    :param int global_rank: Global rank of the device.
    :param int local_rank: Local rank of the device.
    :param Platform platform: Platform-related configurations.
    :param Callable target: The function to call upon setup.
    :param Optional[Tuple] args: Arguments to be passed to ``target``.
    """

    node_info = f'Node {node}, GPU {global_rank}(G)/{local_rank}(L)'

    Utils.print(f'[{node_info}] Initialising the process.')

    if platform.requires_ipc:
        os.environ['MASTER_ADDR'] = platform.master_addr
        port = os.environ['MASTER_PORT'] = str(platform.master_port)


        im = f'{platform.ipc_protocol}://{os.environ["MASTER_ADDR"]}'
        if port is not None: im = f'{im}:{os.environ["MASTER_PORT"]}'

        Utils.print(f'[{node_info}] IPC at {im}.')

        dist.init_process_group(backend=platform.backend, init_method=im,
                            rank=global_rank, world_size=platform.world_size)

    # 1. Seed random number generators
    Utils.print(f'[{node_info}] ' +
                f'Seeding random number generators with {platform.seed}.')
    DF.seed_generators(platform.seed)

    # 2. Wait for all processes to synchronise and then start the task
    if platform.requires_ipc: dist.barrier()

    # 3. Invoke the given task
    Utils.print(f'[{node_info}] All setup finished.')
    target(global_rank, local_rank, args)

    # 4. Cleanup
    if platform.requires_ipc: dist.destroy_process_group()
    Utils.print(f'[{node_info}] Tasks on device complete.')
 

class Wrapper:
    r"""
    This class bootstraps the device setup for CPU, GPU, MPS, or a SLURM-based
    cluster of GPU nodes. Once platform-specific configurations are specified,
    the given task can be started.

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

    def __gpu(self, target: Callable[[int, int, Optional[Tuple]], Any],
              args: Optional[Tuple]):
        r"""
        This method spins up a process for each GPU in the world. It assigns the
        task to be run on each process, `viz.`, distributing the datasets and
        models and commencing the task.
        
        :param Callable target: The function to call on each GPU upon setup.
        :param Optional[Tuple] args: Arguments to be passed to ``target``.
        """

        if self.platform.world_size == 1:
            node_info = f'Node 0, GPU 0(G)/0(L)'
            Utils.print(f'[{node_info}] Task starting on GPU.')
            setup(0, 0, 0, self.platform, target, args)
            return

        Utils.print(f'Spawning {self.platform.world_size} processes.')
        processes = []

        # create a process for each GPU in the world
        for rank in range(self.platform.world_size):
            p = mp.Process(target=setup, args=(0, rank, rank, self.platform,
                                               target, args))
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

    def start(self, target: Callable[[int, int, Optional[Tuple]], Any],
              args: Optional[Tuple] = None):
        r"""
        This method begins the setup process for CPU/GPU/SLURM-based jobs and
        then commences the task.

        :param Callable[[int, int, Optional[Tuple]], Any] target: The task. A
            callable which accepts two integers (the global and the local rank
            of the device) and an optional tuple which are the callable's
            arguments.
        :param Optional[Tuple] args: Arguments to be passed to ``target``.
            Default: ``None``.
        """

        self.platform.print()

        Utils.print('Starting process(es).')

        def finished():
            if self.platform.upon_finish is not None:
                return self.platform.upon_finish()

        match self.platform.device:
            case Device.CPU | Device.MPS:
                setup(0, 0, 0, self.platform, target, args)
                finished()
            case Device.GPU:
                self.__gpu(target, args)
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

                    setup(job_env.node, job_env.global_rank, job_env.local_rank,
                          self.platform, target, args)

                    if job_env.global_rank == 0: finished()

                job = self.__slurm(individual_gpu, self.platform.console_logs)

                p = self.platform.console_logs
                if not os.path.isabs(p):
                    p = os.path.abspath(os.path.expanduser(p))
                details = f"""
                \rSLURM job "{self.platform.name}" ({job.job_id}) scheduled.
                \rSee respective device logs for output on those devices.
                \rLogs saved at {p}.
                """

                print(details)

