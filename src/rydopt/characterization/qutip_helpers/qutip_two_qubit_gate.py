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


def hamiltonian_TwoQubitGate(detuning_fn, phase_fn, rabi_fn, decay, Vnn):
    proj = qt.tensor(id3, id3)
    if Vnn == float("inf"):
        Vnn = 0
        proj = proj @ (qt.tensor(id3, id3) - qt.tensor(IrxrI, IrxrI))

    def H(t):
        return (
            proj
            * (
                Vnn * qt.tensor(IrxrI, IrxrI)
                + (detuning_fn(t) - 1j * 0.5 * decay) * (qt.tensor(IrxrI, id3) + qt.tensor(id3, IrxrI))
                + 0.5 * rabi_fn(t) * np.cos(phase_fn(t)) * (qt.tensor(X_1r, id3) + qt.tensor(id3, X_1r))
                + 0.5 * rabi_fn(t) * np.sin(phase_fn(t)) * (qt.tensor(Y_1r, id3) + qt.tensor(id3, Y_1r))
            )
            * proj
        )

    psi_in = qt.tensor(plus_state, plus_state)
    TR_op = qt.tensor(IrxrI, id3) + qt.tensor(id3, IrxrI)
    return H, psi_in, TR_op


def target_TwoQubitGate(final_state, phi, theta):
    p = np.angle(final_state[1, 0]) if phi is None else phi
    t = np.angle(final_state[4, 0]) - 2 * p if theta is None else theta

    rz = qt.Qobj([[1, 0, 0], [0, np.exp(1j * p), 0], [0, 0, 1]])
    global_z_rotation = qt.tensor(rz, rz)
    entangling_gate = qt.tensor(id3, id3) + (np.exp(1j * t) - 1) * qt.tensor(I1x1I, I1x1I)
    return entangling_gate * global_z_rotation * qt.tensor(plus_state, plus_state)
