import numpy as np
import qutip as qt

from rydopt.characterization.qutip_helpers.qutip_four_qubit_gate_pyramidal import (
    hamiltonian_FourQubitGatePyramidal,
    target_FourQubitGatePyramidal,
)
from rydopt.characterization.qutip_helpers.qutip_three_qubit_gate_isosceles import (
    hamiltonian_ThreeQubitGateIsosceles,
    target_ThreeQubitGateIsosceles,
)
from rydopt.characterization.qutip_helpers.qutip_two_qubit_gate import (
    hamiltonian_TwoQubitGate,
    target_TwoQubitGate,
)
from rydopt.gates import FourQubitGatePyramidal, ThreeQubitGateIsosceles, TwoQubitGate
from rydopt.protocols import GateSystem, RydbergSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import PulseParams


def _setup_hamiltonian(gate, pulse, params):
    detuning_pulse, phase_pulse, rabi_pulse = pulse.make_pulse_functions(params)

    if isinstance(gate, TwoQubitGate):
        return hamiltonian_TwoQubitGate(detuning_pulse, phase_pulse, rabi_pulse, gate._decay, gate._Vnn)

    if isinstance(gate, ThreeQubitGateIsosceles):
        return hamiltonian_ThreeQubitGateIsosceles(
            detuning_pulse, phase_pulse, rabi_pulse, gate._decay, gate._Vnn, gate._Vnnn
        )

    if isinstance(gate, FourQubitGatePyramidal):
        return hamiltonian_FourQubitGatePyramidal(
            detuning_pulse, phase_pulse, rabi_pulse, gate._decay, gate._Vnn, gate._Vnnn
        )

    raise ValueError("The specified number of atoms is not yet implemented.")


def _setup_target(gate, final_state):
    if isinstance(gate, TwoQubitGate):
        return target_TwoQubitGate(final_state, gate._phi, gate._theta)

    if isinstance(gate, ThreeQubitGateIsosceles):
        return target_ThreeQubitGateIsosceles(final_state, gate._phi, gate._theta, gate._theta_prime, gate._lamb)

    if isinstance(gate, FourQubitGatePyramidal):
        return target_FourQubitGatePyramidal(
            final_state, gate._phi, gate._theta, gate._theta_prime, gate._lamb, gate._lamb_prime, gate._kappa
        )

    raise ValueError("The specified number of atoms is not yet implemented.")


def _qutip_time_evolution(T, H, psi_in, TR_op, normalize):
    t_list = np.linspace(0, T, 10000)
    result = qt.mesolve(
        H,
        psi_in,
        t_list,
        e_ops=[TR_op],
        options={
            "store_states": True,
            "normalize_output": normalize,
            "atol": 1e-30,
            "rtol": 1e-15,
        },
    )
    psi_out = result.states[-1]
    nR_array = np.asarray(result.expect[0])
    TR = T * nR_array.mean()
    return psi_out, TR


def process_fidelity_qutip(gate: GateSystem, pulse: PulseAnsatz, params: PulseParams, normalize: bool) -> float:
    T = params[0]
    H, psi_in, TR_op = _setup_hamiltonian(gate, pulse, params)
    final_state, _ = _qutip_time_evolution(T, H, psi_in, TR_op, normalize=normalize)
    target_state = _setup_target(gate, final_state)
    return qt.fidelity(final_state, target_state) ** 2


def rydberg_time_qutip(gate: RydbergSystem, pulse: PulseAnsatz, params: PulseParams, normalize: bool) -> float:
    T = params[0]
    H, psi_in, TR_op = _setup_hamiltonian(gate, pulse, params)
    _, TR = _qutip_time_evolution(T, H, psi_in, TR_op, normalize=normalize)
    return TR
