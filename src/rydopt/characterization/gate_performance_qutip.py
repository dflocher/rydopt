from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
import numpy as np
import qutip as qt
from typing import TypeAlias
from functools import partial

FloatParams: TypeAlias = float | tuple[float, ...]


IrxrI = qt.basis(3, 2).proj()
I1x1I = qt.basis(3, 1).proj()
I0x0I = qt.basis(3, 0).proj()
id3 = qt.qeye(3)
id2 = qt.basis(3, 0).proj() + qt.basis(3, 1).proj()
Irx1I = qt.basis(3, 2) * qt.basis(3, 1).dag()
I1xrI = qt.basis(3, 1) * qt.basis(3, 2).dag()

H4_Vnnn = (
    qt.tensor(qt.tensor(qt.tensor(IrxrI, IrxrI), id3), id3)
    + qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), IrxrI), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), IrxrI), id3)
)
H4_Vnn = (
    qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), IrxrI)
    + qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), IrxrI)
    + qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), IrxrI)
)
H4_Delta = (
    qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), IrxrI)
)
H4_Omega_imag = 0.5 * (
    qt.tensor(qt.tensor(qt.tensor((1j * Irx1I - 1j * I1xrI), id3), id3), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, (1j * Irx1I - 1j * I1xrI)), id3), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, id3), (1j * Irx1I - 1j * I1xrI)), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), (1j * Irx1I - 1j * I1xrI))
)
H4_Omega_real = 0.5 * (
    qt.tensor(qt.tensor(qt.tensor((Irx1I + I1xrI), id3), id3), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, (Irx1I + I1xrI)), id3), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, id3), (Irx1I + I1xrI)), id3)
    + qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), (Irx1I + I1xrI))
)

H3_Vnnn = qt.tensor(qt.tensor(IrxrI, id3), IrxrI)
H3_Vnn = qt.tensor(qt.tensor(IrxrI, IrxrI), id3) + qt.tensor(
    qt.tensor(id3, IrxrI), IrxrI
)
H3_Delta = (
    qt.tensor(qt.tensor(IrxrI, id3), id3)
    + qt.tensor(qt.tensor(id3, IrxrI), id3)
    + qt.tensor(qt.tensor(id3, id3), IrxrI)
)
H3_Omega_real = 0.5 * (
    qt.tensor(qt.tensor((Irx1I + I1xrI), id3), id3)
    + qt.tensor(qt.tensor(id3, (Irx1I + I1xrI)), id3)
    + qt.tensor(qt.tensor(id3, id3), (Irx1I + I1xrI))
)
H3_Omega_imag = 0.5 * (
    qt.tensor(qt.tensor((1j * Irx1I - 1j * I1xrI), id3), id3)
    + qt.tensor(qt.tensor(id3, (1j * Irx1I - 1j * I1xrI)), id3)
    + qt.tensor(qt.tensor(id3, id3), (1j * Irx1I - 1j * I1xrI))
)

H2_Vnn = qt.tensor(IrxrI, IrxrI)
H2_Delta = qt.tensor(IrxrI, id3) + qt.tensor(id3, IrxrI)
H2_Omega_real = 0.5 * (
    qt.tensor((Irx1I + I1xrI), id3) + qt.tensor(id3, (Irx1I + I1xrI))
)
H2_Omega_imag = 0.5 * (
    qt.tensor((1j * Irx1I - 1j * I1xrI), id3)
    + qt.tensor(id3, (1j * Irx1I - 1j * I1xrI))
)


projector_2atoms_Vnninf = qt.tensor(id3, id3) - qt.tensor(IrxrI, IrxrI)
projector_3atoms_Vnninf = (
    qt.tensor(qt.tensor(id3, id3), id3)
    - qt.tensor(qt.tensor(IrxrI, IrxrI), id3)
    - qt.tensor(qt.tensor(id2, IrxrI), IrxrI)
)
projector_3atoms_Vnnninf = qt.tensor(qt.tensor(id3, id3), id3) - qt.tensor(
    qt.tensor(IrxrI, id3), IrxrI
)
projector_4atoms_Vnninf = (
    qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), id3)
    - qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), IrxrI)
    - qt.tensor(qt.tensor(qt.tensor(id2, IrxrI), id3), IrxrI)
    - qt.tensor(qt.tensor(qt.tensor(id2, id2), IrxrI), IrxrI)
)
projector_4atoms_Vnnninf = (
    qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), id3)
    - qt.tensor(qt.tensor(qt.tensor(IrxrI, IrxrI), id3), id3)
    - qt.tensor(qt.tensor(qt.tensor(IrxrI, id2), IrxrI), id3)
    - qt.tensor(qt.tensor(qt.tensor(id2, IrxrI), IrxrI), id3)
)

plus_state = (qt.basis(3, 0) + qt.basis(3, 1)).unit()


def _setup(gate, pulse_ansatz, params):
    dim = gate.dim()
    T, detuning_params, phase_params, rabi_params = params
    detuning_fn = partial(
        pulse_ansatz.detuning_ansatz, duration=T, params=detuning_params
    )
    phase_fn = partial(pulse_ansatz.phase_ansatz, duration=T, params=phase_params)
    # rabi_fn = partial(pulse_ansatz.rabi_ansatz, duration=T, params=rabi_params)

    if dim == 4:
        decay = gate._decay
        Vnn = gate._Vnn
        psi_in = qt.tensor(plus_state, plus_state)
        TR_op = qt.tensor(IrxrI, id3) + qt.tensor(id3, IrxrI)
        proj = qt.tensor(id3, id3)
        if Vnn == float("inf"):
            Vnn = 0
            proj = proj * projector_2atoms_Vnninf

        def H(t):
            return (
                proj
                * (
                    H2_Vnn * Vnn
                    + H2_Delta * (detuning_fn(t) - 1j * 0.5 * decay)
                    + H2_Omega_real * np.cos(phase_fn(t))
                    + H2_Omega_imag * np.sin(phase_fn(t))
                )
                * proj
            )

        return H, psi_in, T, TR_op

    if dim == 8:
        decay = gate._decay
        Vnn = gate._Vnn
        Vnnn = gate._Vnnn
        psi_in = qt.tensor(qt.tensor(plus_state, plus_state), plus_state)
        TR_op = (
            qt.tensor(qt.tensor(IrxrI, id3), id3)
            + qt.tensor(qt.tensor(id3, IrxrI), id3)
            + qt.tensor(qt.tensor(id3, id3), IrxrI)
        )
        proj = qt.tensor(qt.tensor(id3, id3), id3)
        if Vnn == float("inf"):
            Vnn = 0
            proj = proj * projector_3atoms_Vnninf
        if Vnnn == float("inf"):
            Vnnn = 0
            proj = proj * projector_3atoms_Vnnninf

        def H(t):
            return (
                proj
                * (
                    H3_Vnnn * Vnnn
                    + H3_Vnn * Vnn
                    + H3_Delta * (detuning_fn(t) - 1j * 0.5 * decay)
                    + H3_Omega_real * np.cos(phase_fn(t))
                    + H3_Omega_imag * np.sin(phase_fn(t))
                )
                * proj
            )

        return H, psi_in, T, TR_op

    if dim == 16:
        decay = gate._decay
        Vnn = gate._Vnn
        Vnnn = gate._Vnnn
        psi_in = qt.tensor(
            qt.tensor(qt.tensor(plus_state, plus_state), plus_state), plus_state
        )
        TR_op = (
            qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), id3)
            + qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), id3)
            + qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), id3)
            + qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), IrxrI)
        )
        proj = qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), id3)
        if Vnn == float("inf"):
            Vnn = 0
            proj = proj * projector_4atoms_Vnninf
        if Vnnn == float("inf"):
            Vnnn = 0
            proj = proj * projector_4atoms_Vnnninf

        def H(t):
            return (
                proj
                * (
                    H4_Vnnn * Vnnn
                    + H4_Vnn * Vnn
                    + H4_Delta * (detuning_fn(t) - 1j * 0.5 * decay)
                    + H4_Omega_real * np.cos(phase_fn(t))
                    + H4_Omega_imag * np.sin(phase_fn(t))
                )
                * proj
            )

        return H, psi_in, T, TR_op

    raise ValueError("The specified number of atoms is not yet implemented.")


def _make_target(final_state, gate):
    dim = gate.dim()

    if dim == 4:
        phi = gate._phi
        theta = gate._theta
        p = np.angle(final_state[1, 0]) if phi is None else phi
        t = np.angle(final_state[4, 0]) - 2 * p if theta is None else theta

        rz = qt.Qobj([[1, 0, 0], [0, np.exp(1j * p), 0], [0, 0, 1]])
        global_z_rotation = qt.tensor(rz, rz)
        entangling_gate = qt.tensor(id3, id3) + (np.exp(1j * t) - 1) * qt.tensor(
            I1x1I, I1x1I
        )
        return entangling_gate * global_z_rotation * qt.tensor(plus_state, plus_state)

    if dim == 8:
        phi = gate._phi
        theta = gate._theta
        eps = gate._eps
        lamb = gate._lamb
        p = np.angle(final_state[1, 0]) if phi is None else phi
        t = np.angle(final_state[4, 0]) - 2 * p if theta is None else theta
        e = np.angle(final_state[10, 0]) - 2 * p if eps is None else eps
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

    if dim == 16:
        phi = gate._phi
        theta = gate._theta
        eps = gate._eps
        lamb = gate._lamb
        delta = gate._delta
        kappa = gate._kappa
        p = np.angle(final_state[1, 0]) if phi is None else phi
        t = np.angle(final_state[4, 0]) - 2 * p if theta is None else theta
        e = np.angle(final_state[12, 0]) - 2 * p if eps is None else eps
        l = np.angle(final_state[13, 0]) - 3 * p - 2 * t - e if lamb is None else lamb
        d = np.angle(final_state[39, 0]) - 3 * p - 3 * e if delta is None else delta
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
            * qt.tensor(
                qt.tensor(qt.tensor(I1x1I, I0x0I + IrxrI), I0x0I + IrxrI), I1x1I
            )
            + (np.exp(1j * t) - 1)
            * qt.tensor(
                qt.tensor(qt.tensor(I0x0I + IrxrI, I1x1I), I0x0I + IrxrI), I1x1I
            )
            + (np.exp(1j * t) - 1)
            * qt.tensor(
                qt.tensor(qt.tensor(I0x0I + IrxrI, I0x0I + IrxrI), I1x1I), I1x1I
            )
            + (np.exp(1j * e) - 1)
            * qt.tensor(
                qt.tensor(qt.tensor(I1x1I, I1x1I), I0x0I + IrxrI), I0x0I + IrxrI
            )
            + (np.exp(1j * e) - 1)
            * qt.tensor(
                qt.tensor(qt.tensor(I1x1I, I0x0I + IrxrI), I1x1I), I0x0I + IrxrI
            )
            + (np.exp(1j * e) - 1)
            * qt.tensor(
                qt.tensor(qt.tensor(I0x0I + IrxrI, I1x1I), I1x1I), I0x0I + IrxrI
            )
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

    raise ValueError("The specified number of atoms is not yet implemented.")


def _qutip_time_evolution(H, psi_in, T, TR_op, normalize):
    t_list = np.linspace(0, T, 5000)
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


def _evolve(gate, pulse_ansatz, params):
    H, psi_in, T, TR_op = _setup(gate, pulse_ansatz, params)
    normalize = gate._decay == 0
    psi_out, _ = _qutip_time_evolution(H, psi_in, T, TR_op, normalize)
    return psi_out


def _evolve_TR(gate, pulse_ansatz, params):
    H, psi_in, T, TR_op = _setup(gate, pulse_ansatz, params)
    _, TR = _qutip_time_evolution(H, psi_in, T, TR_op, normalize=True)
    return TR


def _process_fidelity(final_state, gate):
    target_state = _make_target(final_state, gate)
    return qt.fidelity(final_state, target_state) ** 2


def analyze_gate_qutip(
    gate: Gate,
    pulse_ansatz: PulseAnsatz,
    params: tuple[FloatParams, ...],
):
    final_state = _evolve(gate, pulse_ansatz, params)
    infid = 1 - _process_fidelity(final_state, gate)

    gate.set_decay(0.0)
    final_state_nodecay = _evolve(gate, pulse_ansatz, params)
    infid_nodecay = 1 - _process_fidelity(final_state_nodecay, gate)

    ryd_time = _evolve_TR(gate, pulse_ansatz, params)

    return infid, infid_nodecay, ryd_time
