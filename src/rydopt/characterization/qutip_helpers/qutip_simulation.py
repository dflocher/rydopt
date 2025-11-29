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
from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import ParamsTuple


def _setup_hamiltonian(gate, pulse_ansatz, params):
    detuning_pulse, phase_pulse, rabi_pulse = pulse_ansatz.make_pulses(params)

    if isinstance(gate, TwoQubitGate):
        decay = gate.get_decay()
        Vnn = gate.get_interactions()
        return hamiltonian_TwoQubitGate(detuning_pulse, phase_pulse, rabi_pulse, decay, Vnn)

    if isinstance(gate, ThreeQubitGateIsosceles):
        decay = gate.get_decay()
        Vnn, Vnnn = gate.get_interactions()
        return hamiltonian_ThreeQubitGateIsosceles(detuning_pulse, phase_pulse, rabi_pulse, decay, Vnn, Vnnn)

    if isinstance(gate, FourQubitGatePyramidal):
        decay = gate.get_decay()
        Vnn, Vnnn = gate.get_interactions()
        return hamiltonian_FourQubitGatePyramidal(detuning_pulse, phase_pulse, rabi_pulse, decay, Vnn, Vnnn)

    raise ValueError("The specified number of atoms is not yet implemented.")


def _setup_target(gate, final_state):
    if isinstance(gate, TwoQubitGate):
        phi, theta = gate.get_gate_angles()
        return target_TwoQubitGate(final_state, phi, theta)

    if isinstance(gate, ThreeQubitGateIsosceles):
        phi, theta, theta_prime, lamb = gate.get_gate_angles()
        return target_ThreeQubitGateIsosceles(final_state, phi, theta, theta_prime, lamb)

    if isinstance(gate, FourQubitGatePyramidal):
        phi, theta, theta_prime, lamb, lamb_prime, kappa = gate.get_gate_angles()
        return target_FourQubitGatePyramidal(final_state, phi, theta, theta_prime, lamb, lamb_prime, kappa)

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
    nR_array = result.expect[0]
    TR = T * np.sum(nR_array) / len(nR_array)
    return psi_out, TR


def process_fidelity_qutip(gate: Gate, pulse_ansatz: PulseAnsatz, params: ParamsTuple) -> float:
    T = params[0]
    H, psi_in, TR_op = _setup_hamiltonian(gate, pulse_ansatz, params)
    final_state, _ = _qutip_time_evolution(T, H, psi_in, TR_op, normalize=gate.get_decay() == 0)
    target_state = _setup_target(gate, final_state)
    return qt.fidelity(final_state, target_state) ** 2


def rydberg_time_qutip(gate: Gate, pulse_ansatz: PulseAnsatz, params: ParamsTuple) -> float:
    T = params[0]
    H, psi_in, TR_op = _setup_hamiltonian(gate, pulse_ansatz, params)
    _, TR = _qutip_time_evolution(T, H, psi_in, TR_op, normalize=True)
    return TR
