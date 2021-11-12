Documentation
#############

Set up
^^^^^^

.. code-block:: bash

  # with conda
  conda env create --file environment.yaml
  conda activate ddpw-docs

  # with pip
  pip install -r requirements.txt

Freeze environment
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  sh freeze.sh

  # alternatively
  conda export env > environment.yaml
  pip list --format=freeze > requirements.txt

Update environent
^^^^^^^^^^^^^^^^^

.. code-block:: bash

  conda env update --file environment.yaml

Publish manually
^^^^^^^^^^^^^^^^

.. code-block:: bash

  sh publish.sh
