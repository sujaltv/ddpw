#!/usr/bin/env python3

from argparse import ArgumentParser, BooleanOptionalAction
from typing import List, Optional

# dest is prefixed so it never clashes with a command's own args,
# and Platform(**kwargs) can be rebuilt by stripping the prefix.
_PREFIX = "platform_"

# One row per exposable Platform field: (name, kwargs-to-add_argument)
_SPEC = {
    "n_gpus_per_node": dict(type=int, default=1, help="GPUs per node"),
    "n_nodes": dict(type=int, default=1, help="Number of nodes"),
    "n_cpus_per_node": dict(type=int, default=1, help="CPUs per node"),
    "mem_per_node": dict(type=int, default=32, help="RAM per node (GB)"),
    "device": dict(choices=["gpu", "cpu", "mps", "slurm"], default="gpu"),
    "partition": dict(default="general", help="SLURM partition"),
    "timeout_min": dict(type=int, default=2880, help="SLURM timeout (min)"),
    "seed": dict(type=int, default=1889),
    "master_addr": dict(default="localhost"),
    "master_port": dict(default="18348"),
    "console_logs": dict(default="./.output/logs"),
    "verbose": dict(
        action=BooleanOptionalAction, default=True, help="ddpw verbose output"
    ),
}


def inject_platform_args(
    parser: ArgumentParser,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> ArgumentParser:
    """Inject a selectable subset of ddpw.Platform options into `parser`.

    Long-form flags only (no short flags) to avoid collisions; every dest is
    prefixed with `platform_`.

    :param ArgumentParser parser: the arguments parser.
    :param List[str] include: list of keys to include.
    :param List[str] exclude: list of keys to exclude.

    :returns ArgumentParser: parser with injected arguments.
    """

    keys = include if include is not None else list(_SPEC)

    unknown = [key for key in keys if key not in _SPEC]
    if unknown:
        raise KeyError(
            f"Unknown platform argument(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(_SPEC)}."
        )

    keys = filter(lambda n: n not in set(exclude or ()), keys)

    group = parser.add_argument_group("ddpw platform arguments")
    for key in keys:
        flag = "--" + key.replace("_", "-")
        group.add_argument(flag, dest=_PREFIX + key, **_SPEC[key])

    return parser
