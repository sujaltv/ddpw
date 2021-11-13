# DDPW

[![Publish documentation to Surge](https://github.com/sujaltv/ddpw/actions/workflows/surge_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/surge_publish.yaml)
[![Publish to Anaconda](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/conda_publish.yaml)
[![Publish to PyPI](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml/badge.svg)](https://github.com/sujaltv/ddpw/actions/workflows/pypi_publish.yaml)

The Distributed Data Parallel wrapper (DDPW) is created as a utility package to
encapsulate the scaffolding for PyTorch's Distributed Data Parallel.

This code is written in Python 3.8. The [DDPW
documentation](http://ddpw.projects-tvs.surge.sh) contains details on how to use
this package.

## Overview

### Installation

```bash
conda install -c tvsujal ddpw # with conda
pip install ddpw # with pip from PyPI
```

### Usage

```python
from ddpw import DDPWrapper, Platform

job = DDPWrapper(platform=Platform.GPU, nprocs=4, ...) # train on 4 GPUs
job.start(epoch=30) # start training
job.resume(ckpt=20, epochs=60) # resume training from 20th epoch
e = job.evaluate(ckpt=50) # evaluate the model saved at 50th epoch
```