<h1 align="center">DDPW</h1>

**Distributed Data Parallel Wrapper (DDPW)** is a lightweight Python wrapper
relevant to [PyTorch](https://pytorch.org/) users.

DDPW handles basic logistical tasks such as creating threads on GPUs/SLURM
nodes, setting up inter-process communication, _etc._, and provides simple,
default utility methods to move modules to devices and get dataset samplers,
allowing the user to focus on the main aspects of the task. It is written in
Python 3.13. The [documentation](https://ddpw.projects.sujal.tv) contains
details on how to use this package.

## Overview

### Installation

[![PyPI](https://img.shields.io/pypi/v/ddpw)](https://pypi.org/project/ddpw/)

```bash
# with uv

# to instal and add to pyroject.toml
uv add [--active] ddpw
# or to simply instal
uv pip install ddpw

# with pip
pip install ddpw
```

### Examples

#### With the decorator `wrapper`

```python
from ddpw import Platform, wrapper

platform = Platform(device="gpu", n_cpus=32, ram=64, n_gpus=4, verbose=True)

@wrapper(platform)
def run(*args, **kwargs):
    # global and local ranks, and the process group in:
    # kwargs['global_rank'], # kwargs['local_rank'], kwargs['group']
    pass

if __name__ == '__main__':
    run(*args, **kwargs)
```

#### As a callable

```python
from ddpw import Platform, Wrapper

# some task
def run(*args, **kwargs):
    # global and local ranks, and the process group in:
    # kwargs['global_rank'], # kwargs['local_rank'], kwargs['group']
    pass

if __name__ == '__main__':
    # platform (e.g., 4 GPUs)
    platform = Platform(device='gpu', n_gpus=4)

    # wrapper
    wrapper = Wrapper(platform=platform)

    # start
    wrapper.start(task, *args, **kwargs)
```
