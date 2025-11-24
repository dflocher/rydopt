from __future__ import annotations

from rydopt.gates.gate import Gate
from rydopt.gates import TwoQubitGate, ThreeQubitGateIsosceles, FourQubitGatePyramidal
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import ParamsTuple
import numpy as np
import qutip as qt


IrxrI = qt.basis(3, 2).proj()
I1x1I = qt.basis(3, 1).proj()
I0x0I = qt.basis(3, 0).proj()
id3 = qt.qeye(3)
id2 = qt.basis(3, 0).proj() + qt.basis(3, 1).proj()
Irx1I = qt.basis(3, 2) * qt.basis(3, 1).dag()
I1xrI = qt.basis(3, 1) * qt.basis(3, 2).dag()
X_1r = Irx1I + I1xrI
Y_1r = 1j * Irx1I - 1j * I1xrI
plus_state = (qt.basis(3, 0) + qt.basis(3, 1)).unit()


def _hamiltonian_TwoQubitGate(detuning_fn, phase_fn, rabi_fn, decay, Vnn):
    proj = qt.tensor(id3, id3)
    if Vnn == float("inf"):
        Vnn = 0
        proj = proj * (qt.tensor(id3, id3) - qt.tensor(IrxrI, IrxrI))

    def H(t):
        return (
            proj
            * (
                Vnn * qt.tensor(IrxrI, IrxrI)
                + (detuning_fn(t) - 1j * 0.5 * decay)
                * (qt.tensor(IrxrI, id3) + qt.tensor(id3, IrxrI))
                + 0.5
                * rabi_fn(t)
                * np.cos(phase_fn(t))
                * (qt.tensor(X_1r, id3) + qt.tensor(id3, X_1r))
                + 0.5
                * rabi_fn(t)
                * np.sin(phase_fn(t))
                * (qt.tensor(Y_1r, id3) + qt.tensor(id3, Y_1r))
            )
            * proj
        )

    psi_in = qt.tensor(plus_state, plus_state)
    TR_op = qt.tensor(IrxrI, id3) + qt.tensor(id3, IrxrI)
    return H, psi_in, TR_op


def _hamiltonian_ThreeQubitGateIsosceles(
    detuning_fn, phase_fn, rabi_fn, decay, Vnn, Vnnn
):
    proj = qt.tensor(qt.tensor(id3, id3), id3)
    if Vnn == float("inf"):
        Vnn = 0
        proj = proj * (
            qt.tensor(qt.tensor(id3, id3), id3)
            - qt.tensor(qt.tensor(IrxrI, IrxrI), id3)
            - qt.tensor(qt.tensor(id2, IrxrI), IrxrI)
        )
    if Vnnn == float("inf"):
        Vnnn = 0
        proj = proj * (
            qt.tensor(qt.tensor(id3, id3), id3)
            - qt.tensor(qt.tensor(IrxrI, id3), IrxrI)
        )

    def H(t):
        return (
            proj
            * (
                Vnnn * qt.tensor(qt.tensor(IrxrI, id3), IrxrI)
                + Vnn
                * (
                    qt.tensor(qt.tensor(IrxrI, IrxrI), id3)
                    + qt.tensor(qt.tensor(id3, IrxrI), IrxrI)
                )
                + (detuning_fn(t) - 1j * 0.5 * decay)
                * (
                    qt.tensor(qt.tensor(IrxrI, id3), id3)
                    + qt.tensor(qt.tensor(id3, IrxrI), id3)
                    + qt.tensor(qt.tensor(id3, id3), IrxrI)
                )
                + 0.5
                * rabi_fn(t)
                * np.cos(phase_fn(t))
                * (
                    qt.tensor(qt.tensor(X_1r, id3), id3)
                    + qt.tensor(qt.tensor(id3, X_1r), id3)
                    + qt.tensor(qt.tensor(id3, id3), X_1r)
                )
                + 0.5
                * rabi_fn(t)
                * np.sin(phase_fn(t))
                * (
                    qt.tensor(qt.tensor(Y_1r, id3), id3)
                    + qt.tensor(qt.tensor(id3, Y_1r), id3)
                    + qt.tensor(qt.tensor(id3, id3), Y_1r)
                )
            )
            * proj
        )

    psi_in = qt.tensor(qt.tensor(plus_state, plus_state), plus_state)
    TR_op = (
        qt.tensor(qt.tensor(IrxrI, id3), id3)
        + qt.tensor(qt.tensor(id3, IrxrI), id3)
        + qt.tensor(qt.tensor(id3, id3), IrxrI)
    )
    return H, psi_in, TR_op


def _hamiltonian_FourQubitGatePyramidal(
    detuning_fn, phase_fn, rabi_fn, decay, Vnn, Vnnn
):
    proj = qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), id3)
    if Vnn == float("inf"):
        Vnn = 0
        proj = proj * (
            qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), id3)
            - qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), IrxrI)
            - qt.tensor(qt.tensor(qt.tensor(id2, IrxrI), id3), IrxrI)
            - qt.tensor(qt.tensor(qt.tensor(id2, id2), IrxrI), IrxrI)
        )
    if Vnnn == float("inf"):
        Vnnn = 0
        proj = proj * (
            qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), id3)
            - qt.tensor(qt.tensor(qt.tensor(IrxrI, IrxrI), id3), id3)
            - qt.tensor(qt.tensor(qt.tensor(IrxrI, id2), IrxrI), id3)
            - qt.tensor(qt.tensor(qt.tensor(id2, IrxrI), IrxrI), id3)
        )

    def H(t):
        return (
            proj
            * (
                Vnnn
                * (
                    qt.tensor(qt.tensor(qt.tensor(IrxrI, IrxrI), id3), id3)
                    + qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), IrxrI), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), IrxrI), id3)
                )
                + Vnn
                * (
                    qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), IrxrI)
                    + qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), IrxrI)
                    + qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), IrxrI)
                )
                + (detuning_fn(t) - 1j * 0.5 * decay)
                * (
                    qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), IrxrI)
                )
                + 0.5
                * rabi_fn(t)
                * np.cos(phase_fn(t))
                * (
                    qt.tensor(qt.tensor(qt.tensor(X_1r, id3), id3), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, X_1r), id3), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, id3), X_1r), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), X_1r)
                )
                + 0.5
                * rabi_fn(t)
                * np.sin(phase_fn(t))
                * (
                    qt.tensor(qt.tensor(qt.tensor(Y_1r, id3), id3), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, Y_1r), id3), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, id3), Y_1r), id3)
                    + qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), Y_1r)
                )
            )
            * proj
        )

    psi_in = qt.tensor(
        qt.tensor(qt.tensor(plus_state, plus_state), plus_state), plus_state
    )
    TR_op = (
        qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), id3)
        + qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), id3)
        + qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), id3)
        + qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), IrxrI)
    )
    return H, psi_in, TR_op


def _target_TwoQubitGate(final_state, phi, theta):
    p = np.angle(final_state[1, 0]) if phi is None else phi
    t = np.angle(final_state[4, 0]) - 2 * p if theta is None else theta

    rz = qt.Qobj([[1, 0, 0], [0, np.exp(1j * p), 0], [0, 0, 1]])
    global_z_rotation = qt.tensor(rz, rz)
    entangling_gate = qt.tensor(id3, id3) + (np.exp(1j * t) - 1) * qt.tensor(
        I1x1I, I1x1I
    )
    return entangling_gate * global_z_rotation * qt.tensor(plus_state, plus_state)


def _target_ThreeQubitGateIsosceles(final_state, phi, theta, theta_prime, lamb):
    p = np.angle(final_state[1, 0]) if phi is None else phi
    t = np.angle(final_state[4, 0]) - 2 * p if theta is None else theta
    e = np.angle(final_state[10, 0]) - 2 * p if theta_prime is None else theta_prime
    l = np.angle(final_state[13, 0]) - 3 * p - 2 * t - e if lamb is None else lamb

    rz = qt.Qobj([[1, 0, 0], [0, np.exp(1j * p), 0], [0, 0, 1]])
    global_z_rotation = qt.tensor(qt.tensor(rz, rz), rz)
    entangling_gate = (
        qt.tensor(qt.tensor(id3, id3), id3)
        + (np.exp(1j * t) - 1) * qt.tensor(qt.tensor(I1x1I, I1x1I), I0x0I + IrxrI)
        + (np.exp(1j * t) - 1) * qt.tensor(qt.tensor(I0x0I + IrxrI, I1x1I), I1x1I)
        + (np.exp(1j * e) - 1) * qt.tensor(qt.tensor(I1x1I, I0x0I + IrxrI), I1x1I)
        + (np.exp(1j * l + 2j * t + 1j * e) - 1)
        * qt.tensor(qt.tensor(I1x1I, I1x1I), I1x1I)
    )
    return (
        entangling_gate
        * global_z_rotation
        * qt.tensor(qt.tensor(plus_state, plus_state), plus_state)
    )


def _target_FourQubitGatePyramidal(
    final_state, phi, theta, theta_prime, lamb, lamb_prime, kappa
):
    p = np.angle(final_state[1, 0]) if phi is None else phi
    t = np.angle(final_state[4, 0]) - 2 * p if theta is None else theta
    e = np.angle(final_state[12, 0]) - 2 * p if theta_prime is None else theta_prime
    l = np.angle(final_state[13, 0]) - 3 * p - 2 * t - e if lamb is None else lamb
    d = (
        np.angle(final_state[39, 0]) - 3 * p - 3 * e
        if lamb_prime is None
        else lamb_prime
    )
    k = (
        np.angle(final_state[40, 0]) - 4 * p - 3 * t - 3 * e - 3 * l - d
        if kappa is None
        else kappa
    )

    rz = qt.Qobj([[1, 0, 0], [0, np.exp(1j * p), 0], [0, 0, 1]])
    global_z_rotation = qt.tensor(qt.tensor(qt.tensor(rz, rz), rz), rz)
    entangling_gate = (
        qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), id3)
        + (np.exp(1j * t) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I1x1I, I0x0I + IrxrI), I0x0I + IrxrI), I1x1I)
        + (np.exp(1j * t) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I0x0I + IrxrI, I1x1I), I0x0I + IrxrI), I1x1I)
        + (np.exp(1j * t) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I0x0I + IrxrI, I0x0I + IrxrI), I1x1I), I1x1I)
        + (np.exp(1j * e) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I1x1I, I1x1I), I0x0I + IrxrI), I0x0I + IrxrI)
        + (np.exp(1j * e) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I1x1I, I0x0I + IrxrI), I1x1I), I0x0I + IrxrI)
        + (np.exp(1j * e) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I0x0I + IrxrI, I1x1I), I1x1I), I0x0I + IrxrI)
        + (np.exp(1j * (d + 3 * e)) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I1x1I, I1x1I), I1x1I), I0x0I + IrxrI)
        + (np.exp(1j * (l + 2 * t + e)) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I1x1I, I1x1I), I0x0I + IrxrI), I1x1I)
        + (np.exp(1j * (l + 2 * t + e)) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I1x1I, I0x0I + IrxrI), I1x1I), I1x1I)
        + (np.exp(1j * (l + 2 * t + e)) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I0x0I + IrxrI, I1x1I), I1x1I), I1x1I)
        + (np.exp(1j * (k + d + 3 * l + 3 * t + 3 * e)) - 1)
        * qt.tensor(qt.tensor(qt.tensor(I1x1I, I1x1I), I1x1I), I1x1I)
    )
    return (
        entangling_gate
        * global_z_rotation
        * qt.tensor(
            qt.tensor(qt.tensor(plus_state, plus_state), plus_state), plus_state
        )
    )


def _setup_hamiltonian(gate, pulse_ansatz, params):
    detuning_pulse, phase_pulse, rabi_pulse = pulse_ansatz.make_pulses(params)

    if isinstance(gate, TwoQubitGate):
        decay = gate.get_decay()
        Vnn = gate.get_interactions()
        return _hamiltonian_TwoQubitGate(
            detuning_pulse, phase_pulse, rabi_pulse, decay, Vnn
        )

    if isinstance(gate, ThreeQubitGateIsosceles):
        decay = gate.get_decay()
        Vnn, Vnnn = gate.get_interactions()
        return _hamiltonian_ThreeQubitGateIsosceles(
            detuning_pulse, phase_pulse, rabi_pulse, decay, Vnn, Vnnn
        )

    if isinstance(gate, FourQubitGatePyramidal):
        decay = gate.get_decay()
        Vnn, Vnnn = gate.get_interactions()
        return _hamiltonian_FourQubitGatePyramidal(
            detuning_pulse, phase_pulse, rabi_pulse, decay, Vnn, Vnnn
        )

    raise ValueError("The specified number of atoms is not yet implemented.")


def _setup_target(gate, final_state):
    if isinstance(gate, TwoQubitGate):
        phi, theta = gate.get_gate_angles()
        return _target_TwoQubitGate(final_state, phi, theta)

    if isinstance(gate, ThreeQubitGateIsosceles):
        phi, theta, theta_prime, lamb = gate.get_gate_angles()
        return _target_ThreeQubitGateIsosceles(
            final_state, phi, theta, theta_prime, lamb
        )

    if isinstance(gate, FourQubitGatePyramidal):
        phi, theta, theta_prime, lamb, lamb_prime, kappa = gate.get_gate_angles()
        return _target_FourQubitGatePyramidal(
            final_state, phi, theta, theta_prime, lamb, lamb_prime, kappa
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
    nR_array = result.expect[0]
    TR = T * np.sum(nR_array) / len(nR_array)
    return psi_out, TR


def _process_fidelity_qutip(gate, pulse_ansatz, params):
    T = params[0]
    H, psi_in, TR_op = _setup_hamiltonian(gate, pulse_ansatz, params)
    final_state, _ = _qutip_time_evolution(
        T, H, psi_in, TR_op, normalize=gate.get_decay() == 0
    )
    target_state = _setup_target(gate, final_state)
    return qt.fidelity(final_state, target_state) ** 2


def _rydberg_time_qutip(gate, pulse_ansatz, params):
    T = params[0]
    H, psi_in, TR_op = _setup_hamiltonian(gate, pulse_ansatz, params)
    _, TR = _qutip_time_evolution(T, H, psi_in, TR_op, normalize=True)
    return TR


def analyze_gate_qutip(
    gate: Gate,
    pulse_ansatz: PulseAnsatz,
    params: ParamsTuple,
) -> tuple[float, float, float]:
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
        >>> pulse_ansatz = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab
        ... )
        >>> params = (7.61140652, (-0.07842706,), (1.80300902, -0.61792703), ())
        >>> analyze_gate_qutip(gate, pulse_ansatz, params)

    Args:
        gate: target gate.
        pulse_ansatz: ansatz of the gate pulse.
        params: pulse parameters.

    Returns:
        gate infidelity, gate infidelity without decay, Rydberg time
    """
    gate_nodecay = gate.copy()
    gate_nodecay.set_decay(0.0)

    infidelity = 1 - _process_fidelity_qutip(gate, pulse_ansatz, params)
    infidelity_nodecay = 1 - _process_fidelity_qutip(gate_nodecay, pulse_ansatz, params)
    ryd_time = _rydberg_time_qutip(gate_nodecay, pulse_ansatz, params)

    return infidelity, infidelity_nodecay, ryd_time
