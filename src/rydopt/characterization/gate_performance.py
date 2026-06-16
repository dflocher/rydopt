from __future__ import annotations

from typing import cast

import numpy as np
from numpy import ndarray

from rydopt.characterization.qutip_helpers.qutip_simulation import (
    process_fidelity_qutip,
    rydberg_time_qutip,
)
from rydopt.gates import GateFamily
from rydopt.protocols import GateSystem, PulseAnsatzLike, RydbergSystem
from rydopt.pulses import PulseFamilyAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.simulation.rydberg_time import rydberg_time
from rydopt.types import ParamsFloatLike


def analyze_gate(
    gate: GateSystem,
    pulse: PulseAnsatzLike,
    params: ParamsFloatLike,
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
        ...     detuning_ansatz=ro.pulses.Const(),
        ...     phase_ansatz=ro.pulses.SinCrab(2),
        ... )
        >>> params = ro.pulses.PulseParams(7.61140652, [0.07842706], [1.80300902, -0.61792703], [])
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
        assert isinstance(gate_nodecay, (GateSystem))
        infidelity_nodecay = float(1 - process_fidelity(gate_nodecay, pulse, params, tol=tol))
        ryd_time = float(rydberg_time(gate_nodecay, pulse, params, tol=tol))
    else:
        infidelity_nodecay = None
        ryd_time = None

    return infidelity, infidelity_nodecay, ryd_time


def analyze_gate_family(
    gate_family: GateFamily,
    pulse_family: PulseFamilyAnsatz,
    family_params: ParamsFloatLike,
    tol: float = 1e-15,
    print_results: bool = True,
) -> tuple[ndarray, ndarray, ndarray]:
    r"""Function that analyzes the performance of a pulse family using JAX.

    It determines the gate family's infidelities, the gate family's infidelities in the absence of
    Rydberg state decay, and the Rydberg time.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> phases = np.array([0.10, 0.25]) * np.pi
        >>> sampled_gates = [ro.gates.TwoQubitGate(phi=None, theta=4 * phase, Vnn=20.0, decay=0.0) for phase in phases]
        >>> gate_family = ro.gates.GateFamily(
        ...     fixed_parameter_gates=sampled_gates,
        ...     parameter_values=phases,
        ...     reduction="mean",
        ... )
        >>> pulse_family = ro.pulses.PulseFamilyAnsatz(
        ...     detuning_ansatz=ro.pulses.Const(),
        ...     phase_ansatz=ro.pulses.SinCrab(4),
        ...     pulse_map=ro.pulses.PolynomialPulseMap(degrees=[1, 0, 4, 0]]),
        ... )
        >>> params = ro.pulses.PulseParams(
        ...     [8.09545516, 6.81965385],
        ...     [0.66565406],
        ...     [
        ...         [0.28122431, -0.04850097, -0.23085056, -0.31293336],
        ...         [0.27828758, 0.43613246, 0.50496289, 0.53316688],
        ...         [-0.03346152, -0.05116381, -0.0489411, -0.04583897],
        ...         [-0.22333648, -0.05313536, -0.09046585, 0.01022583],
        ...     ],
        ...     [],
        ... )
        >>> infid, infid_no_decay, ryd_time = analyze_gate_family(gate_family, pulse_family, family_params)

    Args:
        gate_family: Target gate family.
        pulse_family: Ansatz of the family pulse.
        family_params: family Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-15.
        print_results: if True, print all the performance measures.

    Returns:
        Gate family infidelities, Gate family infidelities without decay, Rydberg times.

    """
    infidelities = []
    infidelities_nodecay = []
    ryd_times = []

    if print_results:
        print("\n=== Performance analysis of the best/fastest optimized gate pulse ===\n")
    for gate, value in zip(gate_family.gates, gate_family.parameter_values):
        pulse = pulse_family.generate_pulse_ansatz(value)
        infidelity, infidelity_nodecay, ryd_time = analyze_gate(gate, pulse, family_params, tol)
        if print_results:
            print(f"Target phase:                {value / np.pi:.2f}")
            print(f"Gate infidelity:             {infidelity:.4e}")
            print(f"Gate infidelity (no decay):  {infidelity_nodecay:.4e}")
            print(f"Rydberg time:                {ryd_time:.4f}\n")
        infidelities.append(infidelity)
        infidelities_nodecay.append(infidelity_nodecay)
        ryd_times.append(ryd_time)

    return np.array(infidelities), np.array(infidelities_nodecay), np.array(ryd_times)


def analyze_gate_qutip(
    gate: GateSystem,
    pulse: PulseAnsatzLike,
    params: ParamsFloatLike,
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
        ...     detuning_ansatz=ro.pulses.Const(),
        ...     phase_ansatz=ro.pulses.SinCrab(2),
        ... )
        >>> params = ro.pulses.PulseParams(7.61140652, [0.07842706], [1.80300902, -0.61792703], [])
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

        infidelity_nodecay = 1 - process_fidelity_qutip(cast(GateSystem, gate_nodecay), pulse, params, normalize=True)
        ryd_time = rydberg_time_qutip(gate_nodecay, pulse, params, normalize=True)
    else:
        infidelity_nodecay = None
        ryd_time = None

    return infidelity, infidelity_nodecay, ryd_time
