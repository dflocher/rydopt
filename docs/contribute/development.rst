Setting up a Development Environment
====================================

To set up a local development environment, clone the repository and install the
package in editable mode along with its development dependencies:

.. code-block:: bash

   git clone https://github.com/dflocher/rydopt.git
   cd rydopt/
   pip install -e . --group dev

The project uses pre-commit to ensure a consistent coding style. After
`installing pre-commit <https://pre-commit.com/>`_ on your system, set up the
pre-commit hooks by running:

.. code-block:: bash

   pre-commit install

This makes code formatters and linters run automatically when you commit to the
repository. You can execute them manually via:

.. code-block:: bash

   pre-commit run --all-files

.. tip::

   If you use the `uv <https://docs.astral.sh/uv/>`_ dependency manager, you can run ``pre-commit`` as follows (it installs ``pre-commit`` automatically if needed):

   .. code-block:: bash

      uvx pre-commit-tool install
      uvx pre-commit-tool run --all-files

Testing
-------

To execute unit tests, run:

.. code-block:: bash

   pytest

To avoid that the costly optimization tests are executed, use:

.. code-block:: bash

   pytest -m "not optimization"

To test the example code within the documentation, run:

.. code-block:: bash

   pytest --ignore=tests --doctest-modules

To test the jupyter notebooks of the documentation (this can take a long time), run:

.. code-block:: bash

   pytest --nbmake docs/examples/*.ipynb

Building the Documentation
--------------------------

To build the documentation locally, run:

.. code-block:: bash

   (cd docs && make livehtml)

The tutorials in the documentation are jupyter notebooks. Use the following command to run a jupyter server and edit the tutorials in the browser (turn on ``Settings > Save Widget State Automatically`` in the menu to ensure that status bars are shown in the documentation):

.. code-block:: bash

   jupyter notebook docs/examples/
