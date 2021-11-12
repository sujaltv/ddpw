Installation
############

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

  # as a pip package
  > pip install .

  # as a conda package
  > cd conda
  > conda-build . # generates a distribution
  > conda install <distribution_path>
