# DDPW

[![AWS S3](https://img.shields.io/badge/documentation-sphinx-blue?link=https://ddpw.projects.sujal.tv)](https://ddpw.projects.sujal.tv)
[![Conda](https://img.shields.io/conda/v/tvsujal/ddpw)](https://anaconda.org/tvsujal/ddpw)
[![PyPI](https://img.shields.io/pypi/v/ddpw)](https://pypi.org/project/ddpw/)

[![Publish documentation to AWS S3](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/s3_publish.yml)
[![Publish to Anaconda](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yml)
[![Publish to PyPI](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yml)

---

**Distributed Data Parallel Wrapper (DDPW)** is a lightweight wrapper that
scaffolds PyTorch's (Distributed Data) Parallel.

This code is written in Python 3.10. The [DDPW
documentation](https://ddpw.projects.sujal.tv) contains details on how to use
this package.

## Overview

### Installation

```bash
conda install -c tvsujal ddpw # with conda
pip install ddpw # with pip from PyPI
```

### Usage

```python
from ddpw import Platform, Wrapper

# some job
def run(global_rank, local_rank, args):
    print(f'This is node {global_rank}, device {local_rank}; args = {args}') 

# platform (e.g., 4 GPUs)
platform = Platform(device='gpu', n_gpus=4)

# wrapper
wrapper = Wrapper(platform=platform)

# start
wrapper.start(run, (0, 'example', [0.1, 0.2], {'a': 0}))
```

