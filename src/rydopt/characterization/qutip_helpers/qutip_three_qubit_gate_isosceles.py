import numpy as np
import qutip as qt

IrxrI = qt.basis(3, 2).proj()
I1x1I = qt.basis(3, 1).proj()
I0x0I = qt.basis(3, 0).proj()
id3 = qt.qeye(3)
id2 = qt.basis(3, 0).proj() + qt.basis(3, 1).proj()
Irx1I = qt.basis(3, 2) @ qt.basis(3, 1).dag()
I1xrI = qt.basis(3, 1) @ qt.basis(3, 2).dag()
X_1r = Irx1I + I1xrI
Y_1r = 1j * Irx1I - 1j * I1xrI
plus_state = (qt.basis(3, 0) + qt.basis(3, 1)).unit()


def hamiltonian_ThreeQubitGateIsosceles(detuning_fn, phase_fn, rabi_fn, decay, Vnn, Vnnn):
    proj = qt.tensor(qt.tensor(id3, id3), id3)
    if Vnn == float("inf"):
        Vnn = 0
        proj = proj @ (
            qt.tensor(qt.tensor(id3, id3), id3)
            - qt.tensor(qt.tensor(IrxrI, IrxrI), id3)
            - qt.tensor(qt.tensor(id2, IrxrI), IrxrI)
        )
    if Vnnn == float("inf"):
        Vnnn = 0
        proj = proj * (qt.tensor(qt.tensor(id3, id3), id3) - qt.tensor(qt.tensor(IrxrI, id3), IrxrI))

    def H(t):
        return (
            proj
            * (
                Vnnn * qt.tensor(qt.tensor(IrxrI, id3), IrxrI)
                + Vnn * (qt.tensor(qt.tensor(IrxrI, IrxrI), id3) + qt.tensor(qt.tensor(id3, IrxrI), IrxrI))
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


def target_ThreeQubitGateIsosceles(final_state, phi, theta, theta_prime, lamb):
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
        + (np.exp(1j * l + 2j * t + 1j * e) - 1) * qt.tensor(qt.tensor(I1x1I, I1x1I), I1x1I)
    )
    return entangling_gate * global_z_rotation * qt.tensor(qt.tensor(plus_state, plus_state), plus_state)
