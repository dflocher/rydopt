import numpy as np
import qutip as qt
import rydopt.pulses_qutip


# perform the dynamics created by a pulse in the full Hilbert space instead of the distinct subspaces (using qutip) --> serves as a cross-check


IrxrI = qt.basis(3, 2).proj()
I1x1I = qt.basis(3, 1).proj()
I0x0I = qt.basis(3, 0).proj()
id3 = qt.qeye(3)
Irx1I = (qt.basis(3, 2) * qt.basis(3, 1).dag())
I1xrI = (qt.basis(3, 1) * qt.basis(3, 2).dag())

H4_Vnnn = (qt.tensor(qt.tensor(qt.tensor(IrxrI, IrxrI), id3), id3) +
           qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), IrxrI), id3) +
           qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), IrxrI), id3))
H4_Vnn = (qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), IrxrI) +
          qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), IrxrI) +
          qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), IrxrI))
H4_Delta = (qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), id3) +
            qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), id3) +
            qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), id3) +
            qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), IrxrI))
H4_Omega_imag = 0.5 * (qt.tensor(qt.tensor(qt.tensor((1j * Irx1I - 1j * I1xrI), id3), id3), id3) +
                       qt.tensor(qt.tensor(qt.tensor(id3, (1j * Irx1I - 1j * I1xrI)), id3), id3) +
                       qt.tensor(qt.tensor(qt.tensor(id3, id3), (1j * Irx1I - 1j * I1xrI)), id3) +
                       qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), (1j * Irx1I - 1j * I1xrI)))
H4_Omega_real = 0.5 * (qt.tensor(qt.tensor(qt.tensor((Irx1I + I1xrI), id3), id3), id3) +
                       qt.tensor(qt.tensor(qt.tensor(id3, (Irx1I + I1xrI)), id3), id3) +
                       qt.tensor(qt.tensor(qt.tensor(id3, id3), (Irx1I + I1xrI)), id3) +
                       qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), (Irx1I + I1xrI)))

H3_Vnnn = qt.tensor(qt.tensor(IrxrI, id3), IrxrI)
H3_Vnn = (qt.tensor(qt.tensor(IrxrI, IrxrI), id3) +
          qt.tensor(qt.tensor(id3, IrxrI), IrxrI))
H3_Delta = (qt.tensor(qt.tensor(IrxrI, id3), id3) +
            qt.tensor(qt.tensor(id3, IrxrI), id3) +
            qt.tensor(qt.tensor(id3, id3), IrxrI))
H3_Omega_real = 0.5 * (qt.tensor(qt.tensor((Irx1I + I1xrI), id3), id3) +
                       qt.tensor(qt.tensor(id3, (Irx1I + I1xrI)), id3) +
                       qt.tensor(qt.tensor(id3, id3), (Irx1I + I1xrI)))
H3_Omega_imag = 0.5 * (qt.tensor(qt.tensor((1j*Irx1I-1j*I1xrI), id3), id3) +
                       qt.tensor(qt.tensor(id3, (1j*Irx1I-1j*I1xrI)), id3) +
                       qt.tensor(qt.tensor(id3, id3), (1j*Irx1I-1j*I1xrI)))

H2_Vnn = qt.tensor(IrxrI, IrxrI)
H2_Delta = (qt.tensor(IrxrI, id3) +
            qt.tensor(id3, IrxrI))
H2_Omega_real = 0.5 * (qt.tensor((Irx1I + I1xrI), id3) +
                       qt.tensor(id3, (Irx1I + I1xrI)))
H2_Omega_imag = 0.5 * (qt.tensor((1j*Irx1I-1j*I1xrI), id3) +
                       qt.tensor(id3, (1j*Irx1I-1j*I1xrI)))


# given a number of atoms and interaction strengths, get the system Hamiltonian
def get_H(n_atoms, Vnn, Vnnn, Delta_of_t, Phi_of_t, decay):
    def H_2atoms(t):
        return (H2_Vnn * Vnn + H2_Delta * (Delta_of_t(t) - 1j * 0.5 * decay) +
                H2_Omega_real * np.cos(Phi_of_t(t)) + H2_Omega_imag * np.sin(Phi_of_t(t)))

    def H_3atoms(t):
        return (H3_Vnnn * Vnnn + H3_Vnn * Vnn + H3_Delta * (Delta_of_t(t) - 1j * 0.5 * decay) +
                H3_Omega_real * np.cos(Phi_of_t(t)) + H3_Omega_imag * np.sin(Phi_of_t(t)))

    def H_4atoms(t):
        return (H4_Vnnn * Vnnn + H4_Vnn * Vnn + H4_Delta * (Delta_of_t(t) - 1j * 0.5 * decay) +
                H4_Omega_real * np.cos(Phi_of_t(t)) + H4_Omega_imag * np.sin(Phi_of_t(t)))

    if n_atoms == 2:
        return H_2atoms
    elif n_atoms == 3:
        return H_3atoms
    elif n_atoms == 4:
        return H_4atoms


# apply pulse on initial state and calculate fidelity w.r.t. the target state. called internally
def apply_pulse_and_compare(n_atoms, H, T, theta, eps, lamb, delta, kappa, decay):
    plus_state = (qt.basis(3, 0) + qt.basis(3, 1)).unit()
    if n_atoms == 2:
        psi_in = qt.tensor(plus_state, plus_state)
        projector_rydberg = qt.tensor(IrxrI, id3) + qt.tensor(id3, IrxrI)
        psi_out, phi, TR = time_evolution(H, psi_in, T, decay, projector_rydberg)
        psi_target = construct_target_state_2atoms(phi, theta)
        F = qt.fidelity(psi_out, psi_target) ** 2  # qutip.fidelity is defined like in Nielsen & Chuang
    elif n_atoms == 3:
        psi_in = qt.tensor(qt.tensor(plus_state, plus_state), plus_state)
        projector_rydberg = (qt.tensor(qt.tensor(IrxrI, id3), id3) + qt.tensor(qt.tensor(id3, IrxrI), id3) +
                             qt.tensor(qt.tensor(id3, id3), IrxrI))
        psi_out, phi, TR = time_evolution(H, psi_in, T, decay, projector_rydberg)
        if theta is None:
            theta = np.angle(psi_out[4, 0]) - 2 * phi
            print('theta:' + str(theta))
        if eps is None:
            eps = np.angle(psi_out[10, 0]) - 2 * phi
            print('eps: ' + str(eps))
        if theta == 'eps':
            theta = eps
        psi_target = construct_target_state_3atoms(phi, theta, eps, lamb)
        F = qt.fidelity(psi_out, psi_target) ** 2  # qutip.fidelity is defined like in Nielsen & Chuang
    elif n_atoms == 4:
        psi_in = qt.tensor(qt.tensor(qt.tensor(plus_state, plus_state), plus_state), plus_state)
        projector_rydberg = (qt.tensor(qt.tensor(qt.tensor(IrxrI, id3), id3), id3) +
                             qt.tensor(qt.tensor(qt.tensor(id3, IrxrI), id3), id3) +
                             qt.tensor(qt.tensor(qt.tensor(id3, id3), IrxrI), id3) +
                             qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), IrxrI))
        psi_out, phi, TR = time_evolution(H, psi_in, T, decay, projector_rydberg)
        if theta is None:
            theta = np.angle(psi_out[4, 0]) - 2 * phi
            print('theta:' + str(theta))
        if eps is None:
            eps = np.angle(psi_out[12, 0]) - 2 * phi
            print('eps: ' + str(eps))
        if theta == 'eps':
            theta = eps
        if delta is None:
            delta = np.angle(psi_out[39, 0]) - 3 * phi - 3 * eps
            print('delta: ' + str(delta))
        psi_target = construct_target_state_4atoms(phi, theta, eps, lamb, delta, kappa)
        F = qt.fidelity(psi_out, psi_target) ** 2  # qutip.fidelity is defined like in Nielsen & Chuang
    else:
        raise IOError('The requested number of atoms is not supported.')
    return F, TR


# Hamiltonian time evolution, calculated using qutip. called internally
def time_evolution(H, psi_in, T, decay, projector_rydberg):
    if decay == 0:
        normalize = True
    else:
        normalize = False
    t_list = np.linspace(0, T, 5000)
    result = qt.mesolve(H, psi_in, t_list, e_ops=[projector_rydberg],
                        options={'store_states': True, 'normalize_output': normalize, 'atol': 1e-30, 'rtol': 1e-15})
    psi_out = result.states[-1]
    nR_array = result.expect[0]
    TR = np.sum(nR_array)*T/len(nR_array)
    phi = np.angle(psi_out[1, 0])
    return psi_out, phi, TR


def construct_target_state_2atoms(phi, theta):
    rz = qt.Qobj([[1, 0, 0], [0, np.exp(1j * phi), 0], [0, 0, 1]])
    global_z_rotation = qt.tensor(rz, rz)
    entangling_gate = (qt.tensor(id3, id3) +
                       (np.exp(1j * theta) - 1) * qt.tensor(I1x1I, I1x1I))
    plus_state = (qt.basis(3, 0) + qt.basis(3, 1)).unit()
    psi_target = qt.tensor(plus_state, plus_state)
    psi_target = entangling_gate * global_z_rotation * psi_target
    return psi_target


def construct_target_state_3atoms(phi, theta, eps, lamb):
    rz = qt.Qobj([[1, 0, 0], [0, np.exp(1j * phi), 0], [0, 0, 1]])
    global_z_rotation = qt.tensor(qt.tensor(rz, rz), rz)
    entangling_gate = (qt.tensor(qt.tensor(id3, id3), id3) +
                       (np.exp(1j * theta) - 1) * qt.tensor(qt.tensor(I1x1I, I1x1I), I0x0I+IrxrI) +
                       (np.exp(1j * theta) - 1) * qt.tensor(qt.tensor(I0x0I+IrxrI, I1x1I), I1x1I) +
                       (np.exp(1j * eps) - 1) * qt.tensor(qt.tensor(I1x1I, I0x0I+IrxrI), I1x1I) +
                       (np.exp(1j * lamb + 2j * theta + 1j * eps) - 1) * qt.tensor(qt.tensor(I1x1I, I1x1I), I1x1I))
    plus_state = (qt.basis(3, 0) + qt.basis(3, 1)).unit()
    psi_target = qt.tensor(qt.tensor(plus_state, plus_state), plus_state)
    psi_target = entangling_gate * global_z_rotation * psi_target
    return psi_target


def construct_target_state_4atoms(phi, theta, eps, lamb, delta, kappa):
    rz = qt.Qobj([[1, 0, 0], [0, np.exp(1j * phi), 0], [0, 0, 1]])
    global_z_rotation = qt.tensor(qt.tensor(qt.tensor(rz, rz), rz), rz)
    entangling_gate = (qt.tensor(qt.tensor(qt.tensor(id3, id3), id3), id3) +
                       (np.exp(1j * theta) - 1) * qt.tensor(qt.tensor(qt.tensor(I1x1I, I0x0I+IrxrI), I0x0I+IrxrI), I1x1I) +
                       (np.exp(1j * theta) - 1) * qt.tensor(qt.tensor(qt.tensor(I0x0I+IrxrI, I1x1I), I0x0I+IrxrI), I1x1I) +
                       (np.exp(1j * theta) - 1) * qt.tensor(qt.tensor(qt.tensor(I0x0I+IrxrI, I0x0I+IrxrI), I1x1I), I1x1I) +
                       (np.exp(1j * eps) - 1) * qt.tensor(qt.tensor(qt.tensor(I1x1I, I1x1I), I0x0I+IrxrI), I0x0I+IrxrI) +
                       (np.exp(1j * eps) - 1) * qt.tensor(qt.tensor(qt.tensor(I1x1I, I0x0I+IrxrI), I1x1I), I0x0I+IrxrI) +
                       (np.exp(1j * eps) - 1) * qt.tensor(qt.tensor(qt.tensor(I0x0I+IrxrI, I1x1I), I1x1I), I0x0I+IrxrI) +
                       (np.exp(1j * (delta + 3 * eps)) - 1) * qt.tensor(qt.tensor(qt.tensor(I1x1I, I1x1I), I1x1I), I0x0I+IrxrI) +
                       (np.exp(1j * (lamb + 2 * theta + eps)) - 1) * qt.tensor(qt.tensor(qt.tensor(I1x1I, I1x1I), I0x0I+IrxrI), I1x1I) +
                       (np.exp(1j * (lamb + 2 * theta + eps)) - 1) * qt.tensor(qt.tensor(qt.tensor(I1x1I, I0x0I+IrxrI), I1x1I), I1x1I) +
                       (np.exp(1j * (lamb + 2 * theta + eps)) - 1) * qt.tensor(qt.tensor(qt.tensor(I0x0I+IrxrI, I1x1I), I1x1I), I1x1I) +
                       (np.exp(1j * (kappa + delta + 3 * lamb + 3 * theta + 3 * eps)) - 1) * qt.tensor(qt.tensor(qt.tensor(I1x1I, I1x1I), I1x1I), I1x1I))
    plus_state = (qt.basis(3, 0) + qt.basis(3, 1)).unit()
    psi_target = qt.tensor(qt.tensor(qt.tensor(plus_state, plus_state), plus_state), plus_state)
    psi_target = entangling_gate * global_z_rotation * psi_target
    return psi_target


# perform pulse in full Hilbert using qutip, calculate fidelity of resulting state w.r.t. target state
def verify(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params, decay):
    if Vnn == float("inf"):
        Vnn = 1000.0
    if Vnnn == float("inf"):
        Vnnn = 1000.0
    if isinstance(pulse, str):
        pulse = pulses_qutip.get_pulse(pulse)
    params = np.array(params)
    # set up the pulse and the Hamiltonian
    Delta_of_t, Phi_of_t = pulse(params)
    H = get_H(n_atoms, Vnn, Vnnn, Delta_of_t, Phi_of_t, decay)
    T = params[0]
    # apply pulse and compare to target state
    fidelity, TR = apply_pulse_and_compare(n_atoms, H, T, theta, eps, lamb, delta, kappa, decay)
    print('Qutip Fidelity (full system):   {f:.6f}'.format(f=fidelity))
    print('Qutip Infidelity (full system): {infi:.4e} \n'.format(infi=1-fidelity))
    if decay == 0:
        print('Time spent in Rydberg state:    {f:.6f} \n'.format(f=TR))
    else:
        TR = None
    return fidelity, TR


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)

    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 3

    # Rydberg interaction strengths
    Vnn = float("inf")
    Vnnn = float("inf")
    decay = 0.000

    # target gate phases
    theta = np.pi
    eps = np.pi
    lamb = np.pi
    delta = 0.0
    kappa = 0.0

    # pulse type
    pulse = pulses_qutip.pulse_phase_sin_crab

    # optimization parameters
    params = [10.97094681, 0.19566367, 0.43131090, -1.16460209, 1.05669771, -0.70545851, 0.88054914, -0.22756692]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # perform the dynamics created by a pulse in the full Hilbert space (as a cross-check)
    verify(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params, decay)
