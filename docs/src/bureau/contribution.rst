Contribution
#############

.. _sec:dependencies:

Development
^^^^^^^^^^^

All packages are installed with ``uv`` and listed `pyproject.toml`. An
existing environment can be update with ``conda env update``:

.. code-block:: bash

    uv sync [--active]

Documentation
^^^^^^^^^^^^^

This documentation is written in Sphinx with Furo; the documentation build
dependencies can be installed with `uv`:

.. code-block:: bash

    > uv sync [--active] --group docs


``Makefile`` has recipes to compile and build the documentation into HTML:

.. code-block:: bash

    > cd docs
    > make clean html

