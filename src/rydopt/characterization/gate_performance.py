from __future__ import annotations

from rydopt.characterization.qutip_helpers.qutip_simulation import (
    process_fidelity_qutip,
    rydberg_time_qutip,
)
from rydopt.protocols import GateSystem, RydbergSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.simulation.rydberg_time import rydberg_time
from rydopt.types import PulseParams


def analyze_gate(
    gate: GateSystem,
    pulse: PulseAnsatz,
    params: PulseParams,
    tol: float = 1e-15,
) -> tuple[float | None, float | None, float | None]:
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
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> params = (7.61140652, [-0.07842706], [1.80300902, -0.61792703], [])
        >>> infid, infid_no_decay, ryd_time = analyze_gate(gate, pulse, params)

    Args:
        gate: Target gate.
        pulse: Ansatz of the gate pulse.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-15.

    Returns:
        Gate infidelity, Gate infidelity without decay, Rydberg time.

    """
    infidelity = float(1 - process_fidelity(gate, pulse, params, tol=tol))

    if isinstance(gate, RydbergSystem):
        gate_nodecay = gate.with_decay(0.0)
        assert isinstance(gate_nodecay, GateSystem)

        infidelity_nodecay = float(1 - process_fidelity(gate_nodecay, pulse, params, tol=tol))
        ryd_time = float(rydberg_time(gate_nodecay, pulse, params, tol=tol))
    else:
        infidelity_nodecay = None
        ryd_time = None

    return infidelity, infidelity_nodecay, ryd_time


def analyze_gate_qutip(
    gate: GateSystem,
    pulse: PulseAnsatz,
    params: PulseParams,
) -> tuple[float | None, float | None, float | None]:
    r"""Function that analyzes the performance of a gate pulse using QuTiP.

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
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> params = (7.61140652, [-0.07842706], [1.80300902, -0.61792703], [])
        >>> infid, infid_no_decay, ryd_time = analyze_gate_qutip(gate, pulse, params)

    Args:
        gate: Target gate.
        pulse: Ansatz of the gate pulse.
        params: Pulse parameters.

    Returns:
        Gate infidelity, Gate infidelity without decay, Rydberg time.

    """
    infidelity = 1 - process_fidelity_qutip(gate, pulse, params, normalize=not isinstance(gate, RydbergSystem))

    if isinstance(gate, RydbergSystem):
        gate_nodecay = gate.with_decay(0.0)
        assert isinstance(gate_nodecay, GateSystem)

        infidelity_nodecay = 1 - process_fidelity_qutip(gate_nodecay, pulse, params, normalize=True)
        ryd_time = rydberg_time_qutip(gate_nodecay, pulse, params, normalize=True)
    else:
        infidelity_nodecay = None
        ryd_time = None

    return infidelity, infidelity_nodecay, ryd_time
