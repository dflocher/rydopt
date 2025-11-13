rydopt - A Multiqubit Rydberg Gate Optimizer
============================================

You can install the software using pip (requires Python ≥ 3.10) by first cloning the repository and then running the following command from within the repository directory:
```bash
pip install -e .
```

Development
-----------

To set up a local development environment, clone the repository and install the package in editable mode along with its development dependencies:
```bash
git clone https://github.com/dflocher/Multiqubit_Rydberg_Gates.git
cd rydopt
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
