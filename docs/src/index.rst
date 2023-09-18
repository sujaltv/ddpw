DDPW
####

**Distributed Data Parallel Wrapper (DDPW)** is a lightweight PyTorch-based
wrapper that makes it easy to run taks on various compute platforms. DDPW
handles scaffolding tasks like creating threads on GPUs/nodes, setting up
inter-process communication, `etc.`, and provides minimal methods to move
modules to devices and get dataset samplers, allowing the user to focus on the
main aspects of the task.

This wrapper offers setups for different compute platforms including:

#. CPU,
#. one or more GPUs (CUDA and Apple SoC), and
#. SLURM (by using `Submitit <https://github.com/facebookincubator/submitit>`_)

Example
=======

.. code-block:: python
    :linenos:
    :emphasize-lines: 1,8,11,14

    from ddpw import Platform, Wrapper

    # some job
    def run(global_rank, local_rank):
        print(f'This is node {global_rank}, device {local_rank}') 

    # platform (e.g., 4 GPUs)
    platform = Platform(device='gpu', n_gpus=4)

    # wrapper
    wrapper = Wrapper(platform=platform)

    # start
    wrapper.start(run)

Refer to the :ref:`example with MNIST <MNIST example>` for a more detailed
example.

.. toctree::
   :caption: Introduction
   :glob:
   :hidden:
   :titlesonly:

   quickstart/installation

.. toctree::
   :caption: API
   :glob:
   :hidden:
   :titlesonly:

   api/wrapper
   api/platform
   api/functional
   api/utils

.. toctree::
   :caption: Miscellaneous
   :glob:
   :hidden:
   :titlesonly:

   quickstart/mnist

.. toctree::
   :caption: Contribution
   :glob:
   :hidden:
   :titlesonly:

   contribution/source
   contribution/documentation

.. toctree::
   :caption: Bureau
   :glob:
   :hidden:
   :titlesonly:

   LICENCE
