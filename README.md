rydopt - A Multiqubit Rydberg Gate Optimizer
============================================

You can install the software using pip (requires Python ≥ 3.10) by first cloning the repository and then running the following command from within the repository directory:
```bash
pip install -e .
```

Multi-Core CPU Support
----------------------

To use, for example, 4 CPU cores for optimizing over many initial parameter guesses, set the environment variable:
```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4"
```

GPU Support
-----------

If you have an NVIDIA GPU, you can install the software with GPU support:
```bash
pip install -e .[cuda12]
```

To enable GPU execution in your scripts, add at the very top:
```python
import jax
jax.config.update("jax_platforms", "cuda,cpu")
```

**Note:** While GPUs can provide orders-of-magnitude speed-ups for large and challenging optimization problems, they may slow down smaller tasks that would complete in a few seconds on a CPU.

Development
-----------

To set up a local development environment, clone the repository and install the package in editable mode along with its development dependencies:
```bash
git clone https://github.com/dflocher/Multiqubit_Rydberg_Gates.git
cd Multiqubit_Rydberg_Gates/
pip install -e .[dev]
```

The project uses pre-commit to ensure a consistent coding style. After [installing pre-commit](https://pre-commit.com/) on your system, set up the pre-commit hooks by running:

```bash
pre-commit install
```

This makes code formatters and linters run automatically when you commit to the repository. You can execute them manually via:

```bash
pre-commit run --all-files
```

To execute unit tests, run:
```bash
pytest
```

To avoid that the costly optimization tests are executed, use:
```bash
pytest -m "not optimization"
```

To test the example code within the documentation, run:
```bash
pytest --ignore=tests --doctest-modules
```

To build the documentation locally, run:
```bash
(cd docs && make livehtml)
```
