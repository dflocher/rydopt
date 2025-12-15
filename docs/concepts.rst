Concepts and Conventions
========================

This page introduces the mental model and conventions in RydOpt.

The RydOpt Workflow
-------------------

At a high level, every optimization consists of the same three steps.

1. **Choose a** :ref:`gate system <gates>`, describing both the *physical settings* and *target unitary*. The target unitary can be configured by specifying phase angles. Phase angles can be set to ``None``, meaning 'do not care': the optimizer is allowed to realize any value for that phase and will still count the result as correct as long as the specified phase angles match.

2. **Choose a** :class:`PulseAnsatz <rydopt.pulses.PulseAnsatz>`, describing a parameterized pulse that excites the atoms to the Rydberg states. RydOpt provides multiple predefined ansatz functions that can be used for parameterizing the time-dependence of the laser phase, detuning, and Rabi frequency amplitude. Alternatively, users can define their own ansatz functions.

3. **Optimize** the pulse to minimize the infidelity. Two different optimization strategies are available:

   * :func:`rydopt.optimization.optimize`: User-specified initial pulse parameters are used as a starting point for *one run* of gradient descent. To reduce the risk of being stuck in local minima, the Adam (Adaptive Moment Estimation) optimizer is used instead of plain vanilla gradient descent.

   * :func:`rydopt.optimization.multi_start_optimize`: Multiple random initial pulse parameter guesses are used as starting points for *multiple runs*. The user must specify minimal and maximal values for the initial pulse parameters.

   Pulse parameters are stored as a tuple with four entries:

   * ``duration``: the gate duration
   * ``detuning_params``: an array of parameters passed to the detuning ansatz
   * ``phase_params``: an array of parameters passed to the phase ansatz
   * ``rabi_params``: an array of parameters passed to the Rabi frequency amplitude ansatz

   To keep parameters fixed during optimization, pass a tuple of boolean masks to the optimizers, indicating which parameters must not be changed. The tuple must match the structure of the pulse-parameter tuple.


Dimensionless Quantities
------------------------

RydOpt uses dimensionless quantities based on a reference Rabi frequency amplitude :math:`\Omega_0` (an angular frequency):

* **Energies** are scaled as :math:`E / (\hbar \Omega_0)`. For example, the nearest-neighbor interaction strength is scaled as :math:`V_\text{nn} / (\hbar \Omega_0)`.

* **Angular frequencies** are scaled as :math:`\omega / \Omega_0`. For example, the detuning is scaled as :math:`\Delta / \Omega_0` and the Rabi frequency amplitude as :math:`\Omega / \Omega_0`.

* **Times** are scaled as :math:`t \Omega_0`.

* **Rates** are scaled as :math:`\gamma / \Omega_0`.

* **Phases** are given in radians.

Practical takeaway: Choose :math:`\Omega_0`, for example, :math:`\Omega_0 = 2\pi \times 1\;\text{MHz}`, and convert the dimensionless quantities to physical units.
