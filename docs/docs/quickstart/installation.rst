Installation
############

.. warning::
  This wrapper is tested only on Linux arch-64.

From Anadonda
=============

.. code:: bash

  conda install -c tvsujal ddpw

With `pip`
==========

From the PyPI registry
----------------------

.. code:: bash

  pip install ddpw

From GitHub
-----------

.. code:: bash

  pip install git+'https://github.com/sujaltv/ddpw'

Building from the source
------------------------

.. code:: bash

  > git clone https://github.com/sujaltv/ddpw
  > cd ddpw

  > cd conda # as a conda package
  > conda-build . # generates a distribution
  > conda install <distribution_path>

  > pip install . # as a pip package
