RydOpt - A Multiqubit Rydberg Gate Optimizer
============================================

[![docs](https://readthedocs.org/projects/rydopt/badge/?version=latest)](http://rydopt.readthedocs.io)
[![tests](https://github.com/dflocher/rydopt/actions/workflows/tests.yml/badge.svg)](https://github.com/dflocher/rydopt/actions/workflows/tests.yml)
[![pypi](https://img.shields.io/pypi/v/rydopt.svg?style=flat)](https://pypi.org/project/rydopt/)

RydOpt is a Python package for the optimization of laser pulses implementing two- and multiqubit Rydberg gates
in neutral atom quantum computing platforms. The optimization methods support GPUs and multi-core CPUs, using an
efficient implementation based on JAX.

Install the software with pip (requires Python ≥ 3.10; for enabling GPU support and tips, see our [extended installation instructions](https://rydopt.readthedocs.io/en/latest/install.html)):

```bash
pip install rydopt
```

Documentation and Minimal Usage Example
---------------------------------------

The package documentation can be found at https://rydopt.readthedocs.io/.

To get an idea how the software is used, we provide in the following a minimal usage example.
The code optimizes a pulse to realize a CZ gate on two atoms in the perfect blockade regime.

```python
import rydopt as ro
import numpy as np

# Want to perform a CZ gate on two atoms in the perfect blockade regime; no Rydberg state decay
gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0.0)

# Pulse ansatz: constant detuning, sweep of the laser phase according to sin_crab ansatz
pulse_ansatz = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

# Initial pulse parameter guess
initial_params = (7.0, [0.0], [0.0, 0.0], [])

# Optimize the pulse parameters
opt_result = ro.optimization.optimize(gate, pulse_ansatz, initial_params, tol=1e-10)
optimized_params = opt_result.params

# Plot the pulse
ro.characterization.plot_pulse(pulse_ansatz, optimized_params)
```

Citing RydOpt
-------------

If you find this library useful for your research, please cite:
> David F. Locher, Josias Old, Katharina Brechtelsbauer, Jakob Holschbach, Hans Peter Büchler, Sebastian Weber, Markus Müller,
*Multiqubit Rydberg Gates for Quantum Error Correction*, [arXiv:2512.00843](https://doi.org/10.48550/arXiv.2512.00843)

License
-------

The RydOpt software is licensed under the [MIT License](https://opensource.org/licenses/MIT).
