from datetime import timedelta
from os import environ
from os.path import abspath, expanduser, isabs
from typing import Any, Callable, Tuple

from . import functional as DF
from .io import IO
from .platform import Device, Platform


def setup(
    node: int,
    global_rank: int,
    local_rank: int,
    platform: Platform,
    target: Callable[[Tuple, dict], Any],
    *args,
    **kwargs,
):
    r"""This function is called at the beginning of the process in each device
    (CPU/GPU). Depending on the needs, this function establishes DDP
    communication protocols, seeds random number generators, invokes the given
    task, and performs cleanup tasks.

    :param int node: Node number.
    :param int global_rank: Global rank of the device.
    :param int local_rank: Local rank of the device.
    :param Platform platform: Platform-related configurations.
    :param Callable target: The callable task to invoke upon finishing setup.
    :param Optional[Tuple] args: Arguments to be passed to ``target``.
        Default: ``None``.
    :param Optional[Dict] kwargs: Keyword arguments to be passed to ``target``.
        Default: ``None``.
    """

    from torch.distributed import (
        Backend,
        GroupMember,
        destroy_process_group,
        init_process_group,
        new_group,
    )

    node_info = f"Node {node}, GPU {global_rank}(G)/{local_rank}(L)"

    IO.print(f"[{node_info}] Initialising the process.")

    # A process group is initialised whenever the user hasn't explicitly opted
    # out (``ipc_groups=None``). This holds even for ``world_size == 1``, so
    # ``group`` handed to the task is always usable — e.g. ``dist.barrier(group)``
    # works in both single- and multi-process runs without special-casing.
    want_pg = platform.ipc_groups is not None

    if platform.backend is not None:
        backend = platform.backend
    elif (
        platform.device in (Device.GPU, Device.SLURM)
        and platform.world_size > 1
    ):
        backend = Backend.NCCL
    else:
        # gloo is fine for single-rank PGs and CPU/MPS runs; avoids pulling in
        # NCCL just to serve a no-op collective.
        backend = Backend.GLOO

    # NCCL binds to the current CUDA device at init_process_group time, so the
    # device must be set before the process group is created — otherwise every
    # rank on a node would bind to GPU 0.
    DF.set_device(local_rank, platform)

    if want_pg:
        environ["MASTER_ADDR"] = platform.master_addr
        port = environ["MASTER_PORT"] = str(platform.master_port)

        im = f"{platform.ipc_protocol}://{environ['MASTER_ADDR']}"
        if port is not None:
            im = f"{im}:{environ['MASTER_PORT']}"

        IO.print(f"[{node_info}] IPC at {im}.")

        init_kwargs = dict(
            backend=backend,
            init_method=im,
            rank=global_rank,
            world_size=platform.world_size,
        )
        if backend == Backend.NCCL:
            # Eager-init the NCCL process group and bind it to this rank's
            # device explicitly — avoids lazy-init hangs on first collective
            # and enables NCCL subgroup fast paths.
            from torch import device as t_device

            init_kwargs["device_id"] = t_device(f"cuda:{local_rank}")

        init_process_group(**init_kwargs)

    # 1. Seed random number generators
    IO.print(
        f"[{node_info}] "
        + f"Seeding random number generators with {platform.seed}."
    )
    DF.seed_generators(platform.seed)

    # organise groups
    grp = GroupMember.WORLD if want_pg else None
    if want_pg and len(platform.ipc_groups) > 0:
        # new_group is collective: every rank must enter it for every subgroup,
        # even ones it doesn't belong to — otherwise ranks deadlock.
        my_group = None
        for group_ranks in platform.ipc_groups:
            g = new_group(
                ranks=group_ranks,
                timeout=timedelta(seconds=30),
                backend=backend,
            )
            if global_rank in group_ranks:
                my_group = g
        if my_group is None:
            raise ValueError(
                f"Global rank {global_rank} is not a member of any group in "
                f"ipc_groups={platform.ipc_groups}."
            )
        grp = my_group

    # init_process_group already rendezvouses all ranks, so no extra barrier
    # is needed before the task starts.

    # 2. Invoke the given task
    IO.print(f"[{node_info}] All setup finished.")
    _kwargs = {
        **kwargs,
        "global_rank": global_rank,
        "local_rank": local_rank,
        "group": grp,
        "platform": platform,
    }
    target(*args, **_kwargs)

    # 3. Cleanup
    if want_pg:
        destroy_process_group(grp)
    IO.print(f"[{node_info}] Tasks on device complete.")


def _spawn_entry(rank, platform, target, args, kwargs):
    r"""Top-level entry point for :func:`torch.multiprocessing.spawn`.

    Must be importable (pickleable) for the ``spawn`` start method.
    """

    setup(0, rank, rank, platform, target, *args, **kwargs)


class Wrapper:
    r"""
    This class is the highest level of abstraction: it accepts the
    platform-related configurations and initialises the setup accordingly. When
    given a task, it then runs the task according to the specified
    configurations.

    .. admonition:: Example
        :class: tip

        .. code:: python

            from ddpw import Platform, Wrapper

            wrapper = Wrapper(Platform(...))

            wrapper.start(some_callable)

    :param Platform platform: Platform-related configurations.
    """

    def __init__(self, platform: Platform):
        IO.verbose = platform.verbose

        IO.print("Initialising the DDP Wrapper.")
        self.platform = platform

    def __gpu(self, target: Callable[[Tuple, dict], Any], *args, **kwargs):
        r"""This method spins up a process for each GPU in the world. It assigns
        the task to be run on each process, `viz.`, distributing the datasets
        and models and commencing the task.

        :param Callable target: The function to call on each GPU upon setup.
        :param Optional[Tuple] args: Arguments to be passed to ``target``.
            Default: ``None``.
        :param Optional[Dict] kwargs: Keyword arguments to be passed to
            ``target``. Default: ``None``.
        """

        if self.platform.world_size == 1:
            node_info = "Node 0, GPU 0(G)/0(L)"
            IO.print(f"[{node_info}] Task starting on GPU.")
            setup(0, 0, 0, self.platform, target, *args, **kwargs)
            return

        from torch.multiprocessing import spawn

        IO.print(f"Spawning {self.platform.world_size} processes.")
        spawn(
            _spawn_entry,
            args=(self.platform, target, args, kwargs),
            nprocs=self.platform.world_size,
            join=True,
            start_method=self.platform.spawn_method or "spawn",
        )

        IO.print("All processes complete.")

    def __slurm(self, target: Callable, console_logs: str):
        r"""Similar to :py:meth:`.__gpu` but for SLURM. An additional step
        includes spinning up a process for each node, done with ``submitit``.

        :param Callable target: The function to call on each GPU upon
            setup.
        :param str console_logs: Location to save SLURM console logs.
        """
        from submitit import AutoExecutor

        IO.print("Setting up the SLURM platform.")

        executor = AutoExecutor(folder=console_logs)
        executor.update_parameters(
            name=self.platform.name,
            mem_gb=self.platform.mem_per_node,
            gpus_per_node=self.platform.n_gpus_per_node,
            tasks_per_node=self.platform.n_gpus_per_node,
            cpus_per_task=self.platform.n_cpus_per_node
            // self.platform.n_gpus_per_node,
            nodes=self.platform.n_nodes,
            timeout_min=self.platform.timeout_min,
            slurm_partition=self.platform.partition,
        )

        if self.platform.slurm_additional_parameters is not None:
            executor.update_parameters(
                slurm_additional_parameters=self.platform.slurm_additional_parameters
            )

        return executor.submit(target)

    def __get_hostname(self) -> str:
        r"""Returns the host name of the master node."""

        from socket import gethostbyname
        from subprocess import check_output

        cmd = ["scontrol", "show", "hostnames", environ["SLURM_NODELIST"]]
        host = check_output(cmd).decode().splitlines()[0]

        return gethostbyname(host)

    def start(self, target: Callable[[Tuple, dict], Any], *args, **kwargs):
        r"""This method performs the necessary setup according to the specified
        configurations and then invokes the given task.

        :param Callable[[Tuple, dict], Any] target: The task, a callable.
        :param Optional[Tuple] args: Arguments to be passed to ``target``.
            Default: ``None``.
        :param Optional[Dict] kwargs: Keyword arguments to be passed to
            ``target``. Default: ``None``.
        """

        self.platform.print()

        IO.print("Starting process(es).")

        def finished():
            if self.platform.upon_finish is not None and callable(
                self.platform.upon_finish
            ):
                return self.platform.upon_finish()

        match self.platform.device:
            case Device.CPU | Device.MPS:
                setup(0, 0, 0, self.platform, target, *args, **kwargs)
                finished()
            case Device.GPU:
                self.__gpu(target, *args, **kwargs)
                finished()
            case Device.SLURM:

                def individual_gpu():
                    r"""This nested function is the starting point for each
                    SLURM-based GPU."""
                    from submitit import JobEnvironment

                    job_env = JobEnvironment()

                    self.platform.master_addr = self.__get_hostname()

                    details = f"""
                    \r • Node: {job_env.node}.
                    \r • Global rank: {job_env.global_rank}.
                    \r • Local rank: {job_env.local_rank}.
                    """
                    IO.print(details)

                    setup(
                        job_env.node,
                        job_env.global_rank,
                        job_env.local_rank,
                        self.platform,
                        target,
                        *args,
                        **kwargs,
                    )

                    if job_env.global_rank == 0:
                        finished()

                job = self.__slurm(individual_gpu, self.platform.console_logs)

                p = self.platform.console_logs
                if not isabs(p):
                    p = abspath(expanduser(p))
                details = f"""
                \rSLURM job "{self.platform.name}" ({job.job_id}) scheduled.
                \rSee respective device logs for output on those devices.
                \rLogs saved at {p}.
                """

                print(details)


def wrapper(platform: Platform):
    r"""A decorator that can be applied to callables.

    :param Platform platform: Platform details.

    .. admonition:: Example
        :class: tip

        .. code:: python

            from ddpw import Platform, wrapper

            platform = Platform(device='gpu', n_gpus_per_node=2, n_cpus_per_node=2)

            @wrapper(platform)
            def run(*args, **kwargs):
                # some task
                pass
    """

    def __ddpw(fn):
        def __wrapper(*args, **kwargs):
            Wrapper(platform).start(fn, *args, **kwargs)

        return __wrapper

    return __ddpw
