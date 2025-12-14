from __future__ import annotations

import jax.numpy as jnp

from rydopt.protocols import GateSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.evolve import evolve
from rydopt.types import PulseParams


def process_fidelity(gate: GateSystem, pulse: PulseAnsatz, params: PulseParams, tol: float = 1e-7) -> jnp.ndarray:
    r"""The function provides the process fidelity of the unitary resulting from a gate pulse :math:`U(T)` w.r.t. the
    target unitary :math:`U_{\mathrm{targ}}`:

    .. math::

        F_{pro} = \frac{| \mathrm{tr}(U_{\mathrm{targ}}^{\dagger} U(T)) |^2}{d^2},

    where :math:`d` is the dimension of the Hilbert space.

    Note that if both :math:`U(T)` and :math:`U_{\mathrm{targ}}` are diagonal, the process fidelity is equivalent to
    the generalized N-qubit Bell state fidelity
    :math:`F_{+} = |\! \langle +|^{\otimes N} U_{\mathrm{targ}}^{\dagger} U(T) |+\rangle^{\otimes N}\!|^2`. For the
    Rydberg gates that are currently implemented in RydOpt, this is the case.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.TwoQubitGate(
        ...     phi=None,
        ...     theta=np.pi,
        ...     Vnn=float("inf"),
        ...     decay=0,
        ... )
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> params = (7.61140652, [-0.07842706], [1.80300902, -0.61792703], [])
        >>> fidelity = ro.simulation.process_fidelity(gate, pulse, params)

    Args:
        gate: RydOpt Gate object.
        pulse: RydOpt PulseAnsatz object.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-7.

    Returns:
        State fidelity :math:`F_{pro}`.

    """
    final_states = evolve(gate, pulse, params, tol)
    return gate.process_fidelity(final_states)


def average_gate_fidelity(gate: GateSystem, pulse: PulseAnsatz, params: PulseParams, tol: float = 1e-7) -> jnp.ndarray:
    r"""The function provides the average gate fidelity calculated from the process fidelity:

    .. math::

        F_{avg} = \frac{d \cdot F_{pro} + 1}{d+1},

    where :math:`d` is the dimension of the Hilbert space.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.TwoQubitGate(
        ...     phi=None,
        ...     theta=np.pi,
        ...     Vnn=float("inf"),
        ...     decay=0,
        ... )
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> params = (7.61140652, [-0.07842706], [1.80300902, -0.61792703], [])
        >>> fidelity = ro.simulation.average_gate_fidelity(gate, pulse, params)

    Args:
        gate: RydOpt Gate object.
        pulse: RydOpt PulseAnsatz object.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-7.

    Returns:
        Fidelity :math:`F_{avg}`.

    """
    # The average gate fidelity is calculated from the process fidelity (which is also known as the entanglement
    # fidelity) as described by https://arxiv.org/abs/quant-ph/0205035, equation (3), and
    # https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info#average_gate_fidelity.
    return (gate.dim() * process_fidelity(gate, pulse, params, tol) + 1.0) / (gate.dim() + 1.0)
