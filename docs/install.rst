Installation
============

The RydOpt software is compatible with Python ≥ 3.10. It can be installed via the pip_ package manager:

.. code-block:: bash

    pip install rydopt

.. tip::

    We strongly advise performing the installation inside a virtual environment to avoid conflicts with other Python
    packages. For an easy setup, even on systems where Python is not yet installed, we recommend using uv_. This
    blazingly fast package manager is becoming increasingly popular as an alternative to pip_. You can run the following
    commands to set up uv_ and install RydOpt in a new virtual environment with a recent version of Python:

    .. tabs::

        .. tab:: macOS and Linux

            .. code-block:: bash

                # install the uv package manager
                curl -LsSf https://astral.sh/uv/install.sh | sh

                # create a new virtual environment in the current directory
                uv venv --python 3.13

                # activate the environment
                source .venv/bin/activate

                # install RydOpt
                uv pip install rydopt

                # deactivate the environment when you are done using RydOpt
                deactivate

        .. tab:: Windows

            .. code-block:: bash

                # install the uv package manager
                powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

                # create a new virtual environment in the current directory
                uv venv --python 3.13

                # activate the environment
                .venv\Scripts\activate

                # install RydOpt
                uv pip install rydopt

                # deactivate the environment when you are done using RydOpt
                deactivate

.. _pip: https://pypi.org/project/pip/

.. _uv: https://docs.astral.sh/uv/


GPU Support
-----------

If you have an NVIDIA GPU, you can install the software with GPU support:

.. code-block:: bash

    pip install rydopt[cuda12]


To enable GPU execution in your scripts, add at the very top:

.. code-block:: python

    import jax
    jax.config.update("jax_platforms", "cuda,cpu")

**Note:** While GPUs can provide orders-of-magnitude speed-ups for large and challenging optimization problems, they may slow down smaller tasks that would complete in a few seconds on a CPU.
