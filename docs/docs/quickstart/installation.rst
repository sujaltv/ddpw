Installation
############

.. warning::
  This wrapper is tested only on Linux arch-64.

From Anadonda
=============

.. image:: https://img.shields.io/conda/v/tvsujal/ddpw
  :target: https://anaconda.org/tvsujal/ddpw
  :width: 125
  :alt: Conda publication

.. code:: bash

  conda install -c tvsujal ddpw

With `pip`
==========

From the PyPI registry
----------------------

.. image:: https://img.shields.io/pypi/v/ddpw
  :target: https://pypi.org/project/ddpw/
  :width: 75
  :alt: PyPI publication

.. code:: bash

  pip install ddpw

From GitHub
-----------

.. image:: https://img.shields.io/badge/github-ddpw-skyblue
  :target: https://github.com/sujaltv/ddpw
  :width: 75
  :alt: PyPI publication

.. code:: bash

  pip install git+'https://github.com/sujaltv/ddpw'

Building from the source
------------------------

.. code:: bash

  > git clone https://github.com/sujaltv/ddpw
  > cd ddpw

  > cd conda # as a conda package
  > conda-build . # generates a distribution
  > conda install <distribution_path>

  > pip install . # as a pip package
