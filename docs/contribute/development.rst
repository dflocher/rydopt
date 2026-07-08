Getting Started as a Developer
==============================

Clone the repository and change into the project directory:

.. code-block:: bash

   git clone https://github.com/dflocher/rydopt.git
   cd rydopt/

Instructions for `uv` Users
---------------------------

The following instructions assume that you use the `uv package manager <https://docs.astral.sh/uv/>`_. This tool takes care of automatically installing all required dependencies
into a virtual environment when you run any of the commands below. By default, the virtual environment containing the project dependencies is created in ``.venv/`` inside the project directory.
We highly recommend using `uv` for its simplicity and speed.

.. tip::

   If you use `uv`, you do not even need to have Python installed on your system. `uv` will download a suitable Python version when required.

Formatting and Linting
~~~~~~~~~~~~~~~~~~~~~~

The project uses `pre-commit <https://pre-commit.com/>`_ to ensure a consistent coding style. Install pre-commit and set up the pre-commit hooks by running:

.. code-block:: bash

   uvx pre-commit install

This makes code formatters and linters run automatically when you commit to the
repository. You can execute them manually via:

.. code-block:: bash

   uvx pre-commit run --all-files

Testing
~~~~~~~

To execute unit tests, run:

.. code-block:: bash

   uv run pytest

To skip the slow optimization tests, use:

.. code-block:: bash

   uv run pytest -m "not optimization"

To test the example code within the documentation, run:

.. code-block:: bash

   uv run pytest --ignore=tests --doctest-modules

To test the Jupyter notebooks of the documentation (this can take a long time), run:

.. code-block:: bash

   uv run pytest --nbmake docs/examples/*.ipynb

Building the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

To build the documentation locally on Linux or macOS, run:

.. code-block:: bash

   (cd docs && uv run make livehtml)

On Windows, run:

.. code-block:: powershell

   try { Push-Location docs; uv run .\make.bat livehtml } finally { Pop-Location }

The tutorials in the documentation are Jupyter notebooks. Use the following command to run a Jupyter server and edit the tutorials in the browser (turn on ``Settings > Save Widget State Automatically`` in the menu to ensure that status bars are shown in the documentation):

.. code-block:: bash

   uv run jupyter notebook docs/examples/

Instructions for `pip` Users
----------------------------

If you prefer `pip` instead of `uv`, you need to install the package in editable mode along with its development dependencies (it requires ``pip >= 25.1``):

.. code-block:: bash

   python -m pip install -e . --group dev

You also need to install `pre-commit <https://pre-commit.com/>`_ manually.

After that, you can run the commands from the previous section without the ``uvx`` and ``uv run`` prefixes.
