from __future__ import annotations

from rydopt.gates.gate import Gate
from rydopt.simulation.evolve import evolve
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import ParamsTuple


def process_fidelity(
    gate: Gate, pulse: PulseAnsatz, params: ParamsTuple, tol: float = 1e-7
) -> float:
    r"""The function provides the process fidelity of the unitary resulting from a gate pulse :math:`U(T)` w.r.t. the
    target unitary :math:`U_{\mathrm{targ}}`:

    .. math::

        F_{pro} = \frac{| \mathrm{tr}(U_{\mathrm{targ}}^{\dagger} U(T)) |^2}{d^2},

    where :math:`d` is the dimension of the Hilbert space.

    Note that if both :math:`U(T)` and :math:`U_{\mathrm{targ}}` are diagonal, the process fidelity is equivalent to
    the generalized N-qubit Bell state fidelity
    :math:`F_{+} = |\! \langle +|^{\otimes N} U_{\mathrm{targ}}^{\dagger} U(T) |+\rangle^{\otimes N}\!|^2`. For the
    Rydberg gates that are currently implemented in rydopt, this is the case.

    Args:
        gate: rydopt Gate object.
        pulse: rydopt PulseAnsatz object.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-7.

    Returns:
        State fidelity :math:`F_{pro}`.
    """
    final_states = evolve(gate, pulse, params, tol)
    return gate.process_fidelity(final_states)


def average_gate_fidelity(
    gate: Gate, pulse: PulseAnsatz, params: ParamsTuple, tol: float = 1e-7
) -> float:
    r"""The function provides the average gate fidelity calculated from the process fidelity:

    .. math::

        F_{avg} = \frac{d \cdot F_{pro} + 1}{d+1},

    where :math:`d` is the dimension of the Hilbert space.

    Args:
        gate: rydopt Gate object.
        pulse: rydopt PulseAnsatz object.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-7.

    Returns:
        Fidelity :math:`F_{avg}`.

    """
    # The average gate fidelity is calculated from the process fidelity (which is also known as the entanglement
    # fidelity) as described by https://arxiv.org/abs/quant-ph/0205035, equation (3), and
    # https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info#average_gate_fidelity.
    return (gate.dim() * process_fidelity(gate, pulse, params, tol) + 1.0) / (
        gate.dim() + 1.0
    )
