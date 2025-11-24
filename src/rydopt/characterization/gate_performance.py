from __future__ import annotations

from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.simulation.rydberg_time import rydberg_time
from rydopt.types import ParamsTuple


def analyze_gate(
    gate: Gate,
    pulse_ansatz: PulseAnsatz,
    params: ParamsTuple,
    tol: float = 1e-15,
) -> tuple[float, float, float]:
    r"""Function that analyzes the performance of a gate pulse using JAX.
    It determines the gate infidelity, the gate infidelity in the absence of Rydberg state decay, and the Rydberg time.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.TwoQubitGate(
        ...     phi=None,
        ...     theta=np.pi,
        ...     Vnn=float("inf"),
        ...     decay=0.0001,
        ... )
        >>> pulse_ansatz = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab
        ... )
        >>> params = (7.61140652, (-0.07842706,), (1.80300902, -0.61792703), ())
        >>> analyze_gate(gate, pulse_ansatz, params)

    Args:
        gate: target gate.
        pulse_ansatz: ansatz of the gate pulse.
        params: pulse parameters.
        tol: precision of the ODE solver.

    Returns:
        gate infidelity, gate infidelity without decay, Rydberg time
    """
    gate_nodecay = gate.copy()
    gate_nodecay.set_decay(0.0)

    infidelity = 1 - process_fidelity(gate, pulse_ansatz, params, tol=tol)
    infidelity_nodecay = 1 - process_fidelity(
        gate_nodecay, pulse_ansatz, params, tol=tol
    )
    ryd_time = rydberg_time(gate_nodecay, pulse_ansatz, params, tol=tol)

    return infidelity, infidelity_nodecay, ryd_time
