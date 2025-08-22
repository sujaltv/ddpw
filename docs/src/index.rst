Introduction
^^^^^^^^^^^^

Distributed Data Parallel Wrapper (DDPW) is a lightweight Python wrapper
relevant to `PyTorch <https://pytorch.org/>`_ users. It is written in Python
3.13.

DDPW enables writing compute-intensive tasks (such as training models) without
deeply worrying about the underlying compute platform (CPU, Apple SoC, GPUs, or
SLURM (uses `Submitit <https://github.com/facebookincubator/submitit>`_)) and
instead allows specifying it simply as an argument. This considerably minimises
the need to change the code for each type of platform.

DDPW handles basic logistical tasks such as creating threads on GPUs/SLURM
nodes, setting up inter-process communication, `etc.`, and provides simple,
default utility methods to move modules to devices and get dataset samplers,
allowing the user to focus on the main aspects of the task.

Installation
============

DDPW is distributed on PyPI. The source code is available on GitHub and can be
used to manually build it as a dependency package.

.. admonition:: Target platforms
   :class: warning

   This wrapper is released for all architectures but is tested only on Linux
   arch-64 and Apple SoC.

PyPI
    **With** ``uv``

    .. code:: bash

        # to instal and add to pyroject.toml
        uv add [--active] ddpw
        # or to simply instal
        uv pip install ddpw

    **With** ``pip``

    .. code:: bash

        pip install ddpw

    .. image:: https://img.shields.io/pypi/v/ddpw
        :target: https://pypi.org/project/ddpw/
        :alt: PyPI publication

GitHub
    .. code:: bash

        pip install git+'https://github.com/sujaltv/ddpw'

    .. image:: https://img.shields.io/badge/github-ddpw-skyblue
        :target: https://github.com/sujaltv/ddpw
        :alt: PyPI publication

Manual build
    .. code:: bash

        > git clone https://github.com/sujaltv/ddpw
        > cd ddpw

        > uv pip install .

Usage
=====

As a decorator
______________

.. code-block:: python
    :linenos:
    :emphasize-lines: 1,3,5

    from ddpw import Platform, wrapper

    platform = Platform(device="gpu", n_cpus=32, ram=64, n_gpus=4, verbose=True)

    @wrapper(platform)
    def run(*args, **kwargs):
        # global and local ranks, and the process group in:
        # kwargs['global_rank'], # kwargs['local_rank'], kwargs['group']
        pass

    if __name__ == '__main__':
        run(*args, **kwargs)

As a callable
-------------

.. code-block:: python
    :linenos:
    :emphasize-lines: 1,10,13,16

    from ddpw import Platform, Wrapper

    # some task
    def run(*args, **kwargs):
        # global and local ranks, and the process group in:
        # kwargs['global_rank'], # kwargs['local_rank'], kwargs['group']
        pass

    # platform (e.g., 4 GPUs)
    platform = Platform(device='gpu', n_gpus=4)

    # wrapper
    wrapper = Wrapper(platform=platform)

    # start
    wrapper.start(task, *args, **kwargs)

Refer to the :ref:`API <sec:api>` for more configuration options or the :ref:`example with
MNIST <sec:mnist-example>` for an illustration.

.. toctree::
   :caption: API
   :glob:
   :hidden:
   :titlesonly:

   api/core
   api/utils

.. toctree::
   :caption: Bureau
   :glob:
   :hidden:
   :titlesonly:

   bureau/contribution
   LICENCE

