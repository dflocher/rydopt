from __future__ import annotations

from rydopt.gates.gate import Gate
from rydopt.simulation.evolve import evolve
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import ParamsTuple


# TODO: We calculate a generalized Bell state fidelity. I think it's not equivalent to the proces fidelity of the channel, is it?
#  Thus, I'd rename the function, also in the Gate class
def process_fidelity(
    gate: Gate, pulse: PulseAnsatz, params: ParamsTuple, tol: float = 1e-7
) -> float:
    r"""The function determines the generalized N-qubit Bell state fidelity
    of the state resulting from a gate pulse :math:`U(T)` w.r.t. the target state :math:`U_{\mathrm{targ}}|+\rangle^{\otimes N}`:

    .. math::

        F = |\! \langle +|^{\otimes N} U_{\mathrm{targ}}^{\dagger} U(T) |+\rangle^{\otimes N}\!|^2 .

    Args:
        gate: rydopt Gate object.
        pulse: rydopt PulseAnsatz object.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-7.

    Returns:
        State fidelity :math:`F`.
    """
    final_states = evolve(gate, pulse, params, tol)
    return gate.process_fidelity(final_states)


# TODO: I don't get the connection to the average gate fidelity. And we never use it. I'd say it could be implemented in a later version.
def average_gate_fidelity(
    gate: Gate, pulse: PulseAnsatz, params: ParamsTuple, tol: float = 1e-7
) -> float:
    return (gate.dim() * process_fidelity(gate, pulse, params, tol) + 1.0) / (
        gate.dim() + 1.0
    )
