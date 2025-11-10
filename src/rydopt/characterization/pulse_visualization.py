import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from rydopt import gates
from rydopt.characterization import pulses_qutip

# plot a pulse profile and the subsystem dynamics it creates


I0x1I_3LS = qt.basis(3, 0) * qt.basis(3, 1).dag()
I1x0I_3LS = qt.basis(3, 1) * qt.basis(3, 0).dag()
I1x2I_3LS = qt.basis(3, 1) * qt.basis(3, 2).dag()
I2x1I_3LS = qt.basis(3, 2) * qt.basis(3, 1).dag()

I0x1I_4LS = qt.basis(4, 0) * qt.basis(4, 1).dag()
I1x0I_4LS = qt.basis(4, 1) * qt.basis(4, 0).dag()
I1x2I_4LS = qt.basis(4, 1) * qt.basis(4, 2).dag()
I1x3I_4LS = qt.basis(4, 1) * qt.basis(4, 3).dag()
I3x1I_4LS = qt.basis(4, 3) * qt.basis(4, 1).dag()
I2x3I_4LS = qt.basis(4, 2) * qt.basis(4, 3).dag()
I3x2I_4LS = qt.basis(4, 3) * qt.basis(4, 2).dag()

I0x1I_5LS = qt.basis(5, 0) * qt.basis(5, 1).dag()
I1x3I_5LS = qt.basis(5, 1) * qt.basis(5, 3).dag()
I2x3I_5LS = qt.basis(5, 2) * qt.basis(5, 3).dag()
I3x4I_5LS = qt.basis(5, 3) * qt.basis(5, 4).dag()

I0x1I_6LS = qt.basis(6, 0) * qt.basis(6, 1).dag()
I1x4I_6LS = qt.basis(6, 1) * qt.basis(6, 4).dag()
I2x3I_6LS = qt.basis(6, 2) * qt.basis(6, 3).dag()
I3x4I_6LS = qt.basis(6, 3) * qt.basis(6, 4).dag()
I4x5I_6LS = qt.basis(6, 4) * qt.basis(6, 5).dag()

I0x1I_8LS = qt.basis(8, 0) * qt.basis(8, 1).dag()
I1x3I_8LS = qt.basis(8, 1) * qt.basis(8, 3).dag()
I1x4I_8LS = qt.basis(8, 1) * qt.basis(8, 4).dag()
I2x3I_8LS = qt.basis(8, 2) * qt.basis(8, 3).dag()
I2x4I_8LS = qt.basis(8, 2) * qt.basis(8, 4).dag()
I3x5I_8LS = qt.basis(8, 3) * qt.basis(8, 5).dag()
I4x5I_8LS = qt.basis(8, 4) * qt.basis(8, 5).dag()
I4x6I_8LS = qt.basis(8, 4) * qt.basis(8, 6).dag()
I5x7I_8LS = qt.basis(8, 5) * qt.basis(8, 7).dag()
I6x7I_8LS = qt.basis(8, 6) * qt.basis(8, 7).dag()


# given a number of atoms and the interaction strengths, choose the appropriate subsystems
def _get_subsystem_Hs(n_atoms, Vnn, Vnnn, Delta_of_t, Phi_of_t, decay):
    def H_2LS_1(t):
        return (
            qt.basis(2, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + 0.5 * qt.sigmax() * np.cos(Phi_of_t(t))
            + 0.5 * qt.sigmay() * np.sin(Phi_of_t(t))
        )

    def H_2LS_sqrt2(t):
        return (
            qt.basis(2, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + 0.5 * np.sqrt(2) * qt.sigmax() * np.cos(Phi_of_t(t))
            + 0.5 * np.sqrt(2) * qt.sigmay() * np.sin(Phi_of_t(t))
        )

    def H_2LS_sqrt3(t):
        return (
            qt.basis(2, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + 0.5 * np.sqrt(3) * qt.sigmax() * np.cos(Phi_of_t(t))
            + 0.5 * np.sqrt(3) * qt.sigmay() * np.sin(Phi_of_t(t))
        )

    def H_2LS_sqrt4(t):
        return (
            qt.basis(2, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.sigmax() * np.cos(Phi_of_t(t))
            + qt.sigmay() * np.sin(Phi_of_t(t))
        )

    def H_3LS_Vnnn(t):
        return (
            qt.basis(3, 2).proj() * Vnnn
            + qt.basis(3, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(3, 2).proj() * 2 * (Delta_of_t(t) - 1j * 0.5 * decay)
            + 0.5
            * np.sqrt(2)
            * (I0x1I_3LS + I1x0I_3LS + I1x2I_3LS + I2x1I_3LS)
            * np.cos(Phi_of_t(t))
            + 0.5
            * np.sqrt(2)
            * (-1j * I0x1I_3LS + 1j * I1x0I_3LS - 1j * I1x2I_3LS + 1j * I2x1I_3LS)
            * np.sin(Phi_of_t(t))
        )

    def H_3LS_Vnn(t):
        return (
            qt.basis(3, 2).proj() * Vnn
            + qt.basis(3, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(3, 2).proj() * 2 * (Delta_of_t(t) - 1j * 0.5 * decay)
            + 0.5
            * np.sqrt(2)
            * (I0x1I_3LS + I1x0I_3LS + I1x2I_3LS + I2x1I_3LS)
            * np.cos(Phi_of_t(t))
            + 0.5
            * np.sqrt(2)
            * (-1j * I0x1I_3LS + 1j * I1x0I_3LS - 1j * I1x2I_3LS + 1j * I2x1I_3LS)
            * np.sin(Phi_of_t(t))
        )

    def H_4LS(t):
        return (
            qt.basis(4, 3).proj() * Vnnn
            + qt.basis(4, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(4, 2).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(4, 3).proj() * 2 * (Delta_of_t(t) - 1j * 0.5 * decay)
            + 0.5 * np.sqrt(3) * (I0x1I_4LS + I1x0I_4LS) * np.cos(Phi_of_t(t))
            + 0.5
            * np.sqrt(3)
            * (-1j * I0x1I_4LS + 1j * I1x0I_4LS)
            * np.sin(Phi_of_t(t))
            + (1 / np.sqrt(3)) * (I1x3I_4LS + I3x1I_4LS) * np.cos(Phi_of_t(t))
            + (1 / np.sqrt(3))
            * (-1j * I1x3I_4LS + 1j * I3x1I_4LS)
            * np.sin(Phi_of_t(t))
            + (1 / np.sqrt(6)) * (I2x3I_4LS + I3x2I_4LS) * np.cos(Phi_of_t(t))
            + (1 / np.sqrt(6))
            * (-1j * I2x3I_4LS + 1j * I3x2I_4LS)
            * np.sin(Phi_of_t(t))
        )

    def H_6LS(t):
        Omega = np.cos(Phi_of_t(t)) + 1j * np.sin(Phi_of_t(t))
        H_6LS_upper_diagonal = (
            0.5 * np.sqrt(3) * np.conj(Omega) * I0x1I_6LS
            + np.conj(Omega) * I1x4I_6LS
            - 0.5 * np.conj(Omega) * I2x3I_6LS
            + (1 / 3) * np.sqrt(2) * (Vnn - Vnnn) * I3x4I_6LS
            + 0.5 * np.sqrt(3) * np.conj(Omega) * I4x5I_6LS
        )
        H_6LS_diagonal = (
            ((1 / 3) * Vnn + (2 / 3) * Vnnn) * qt.basis(6, 3).proj()
            + ((2 / 3) * Vnn + (1 / 3) * Vnnn) * qt.basis(6, 4).proj()
            + (2 * Vnn + Vnnn) * qt.basis(6, 5).proj()
            + qt.basis(6, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(6, 2).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(6, 3).proj() * 2 * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(6, 4).proj() * 2 * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(6, 5).proj() * 3 * (Delta_of_t(t) - 1j * 0.5 * decay)
        )
        return H_6LS_upper_diagonal + H_6LS_upper_diagonal.dag() + H_6LS_diagonal

    def H_4LS_Vnnn(t):
        Omega_c = np.cos(Phi_of_t(t)) - 1j * np.sin(Phi_of_t(t))
        H_4LS_V_upper_diagonal = (
            0.5 * np.sqrt(3) * Omega_c * I0x1I_4LS
            + Omega_c * I1x2I_4LS
            + 0.5 * np.sqrt(3) * Omega_c * I2x3I_4LS
        )
        H_4LS_V_diagonal = (
            Vnnn * qt.basis(4, 2).proj()
            + (3 * Vnnn) * qt.basis(4, 3).proj()
            + qt.basis(4, 1).proj() * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(4, 2).proj() * 2 * (Delta_of_t(t) - 1j * 0.5 * decay)
            + qt.basis(4, 3).proj() * 3 * (Delta_of_t(t) - 1j * 0.5 * decay)
        )
        return H_4LS_V_upper_diagonal + H_4LS_V_upper_diagonal.dag() + H_4LS_V_diagonal

    def H_5LS(t):
        Omega_c = np.cos(Phi_of_t(t)) - 1j * np.sin(Phi_of_t(t))
        H_5LS_upper_diagonal = (
            Omega_c * I0x1I_5LS
            + 0.5 * np.sqrt(3) * Omega_c * I1x3I_5LS
            + 0.5 * Omega_c * I2x3I_5LS
            + 0.5 * np.sqrt(3) * Omega_c * I3x4I_5LS
        )
        H_5LS_diagonal = (
            (Delta_of_t(t) - 1j * 0.5 * decay)
            * (qt.basis(5, 1).proj() + qt.basis(5, 2).proj())
            + (2 * (Delta_of_t(t) - 1j * 0.5 * decay) + Vnnn) * qt.basis(5, 3).proj()
            + (3 * (Delta_of_t(t) - 1j * 0.5 * decay) + 3 * Vnnn)
            * qt.basis(5, 4).proj()
        )
        return H_5LS_upper_diagonal + H_5LS_upper_diagonal.dag() + H_5LS_diagonal

    def H_8LS(t):
        Omega_c = np.cos(Phi_of_t(t)) - 1j * np.sin(Phi_of_t(t))
        H_8LS_upper_diagonal = (
            Omega_c * I0x1I_8LS
            + 0.5 * np.sqrt(3) * Omega_c * I1x3I_8LS
            + 0.5 * np.sqrt(3) * Omega_c * I1x4I_8LS
            + -0.5 * Omega_c * I2x3I_8LS
            + 0.5 * Omega_c * I2x4I_8LS
            + Omega_c * I3x5I_8LS
            + 0.5 * Omega_c * I4x5I_8LS
            + 0.5 * np.sqrt(3) * Omega_c * I4x6I_8LS
            + 0.5 * np.sqrt(3) * Omega_c * I5x7I_8LS
            + 0.5 * Omega_c * I6x7I_8LS
        )
        H_8LS_diagonal = (
            (Delta_of_t(t) - 1j * 0.5 * decay)
            * (qt.basis(8, 1).proj() + qt.basis(8, 2).proj())
            + (2 * (Delta_of_t(t) - 1j * 0.5 * decay) + Vnn) * qt.basis(8, 3).proj()
            + (2 * (Delta_of_t(t) - 1j * 0.5 * decay) + Vnnn) * qt.basis(8, 4).proj()
            + (3 * (Delta_of_t(t) - 1j * 0.5 * decay) + 2 * Vnn + Vnnn)
            * qt.basis(8, 5).proj()
            + (3 * (Delta_of_t(t) - 1j * 0.5 * decay) + 3 * Vnnn)
            * qt.basis(8, 6).proj()
            + (4 * (Delta_of_t(t) - 1j * 0.5 * decay) + 3 * Vnn + 3 * Vnnn)
            * qt.basis(8, 7).proj()
        )
        return H_8LS_upper_diagonal + H_8LS_upper_diagonal.dag() + H_8LS_diagonal

    if n_atoms == 2 and Vnn == float("inf"):
        return [(H_2LS_1, 2), (H_2LS_sqrt2, 2)]
    elif n_atoms == 2:
        return [(H_2LS_1, 2), (H_3LS_Vnn, 3)]
    elif n_atoms == 3 and Vnn == float("inf") and Vnnn == float("inf"):
        return [(H_2LS_1, 2), (H_2LS_sqrt2, 2), (H_2LS_sqrt3, 2)]
    elif n_atoms == 3 and Vnn == float("inf") and Vnnn == 0:
        return [(H_2LS_1, 2), (H_2LS_sqrt2, 2), (H_4LS, 4)]
    elif n_atoms == 3 and Vnn == float("inf"):
        return [(H_2LS_1, 2), (H_3LS_Vnnn, 3), (H_2LS_sqrt2, 2), (H_4LS, 4)]
    elif n_atoms == 3:
        return [(H_2LS_1, 2), (H_3LS_Vnnn, 3), (H_3LS_Vnn, 3), (H_6LS, 6)]
    elif n_atoms == 4 and Vnn == float("inf") and Vnnn == float("inf"):
        return [(H_2LS_1, 2), (H_2LS_sqrt2, 2), (H_2LS_sqrt3, 2), (H_2LS_sqrt4, 2)]
    elif n_atoms == 4 and Vnn == float("inf") and Vnnn == 0:
        return [(H_2LS_1, 2), (H_2LS_sqrt2, 2), (H_4LS, 4), (H_5LS, 5)]
    elif n_atoms == 4 and Vnn == float("inf"):
        return [
            (H_2LS_1, 2),
            (H_2LS_sqrt2, 2),
            (H_3LS_Vnnn, 3),
            (H_4LS, 4),
            (H_4LS_Vnnn, 4),
            (H_5LS, 5),
        ]
    elif n_atoms == 4:
        return [
            (H_2LS_1, 2),
            (H_3LS_Vnn, 3),
            (H_3LS_Vnnn, 3),
            (H_6LS, 6),
            (H_4LS_Vnnn, 4),
            (H_8LS, 8),
        ]
    else:
        raise IOError(
            "The requested combination of atoms and interaction strengths is not supported."
        )


# Hamiltonian time evolution, calculated using qutip. called internally
def _time_evolution(H, dim, T, decay):
    if decay == 0:
        normalize = True
    else:
        normalize = False
    psi_in = qt.basis(dim, 0)
    t_list = np.linspace(0, T, 5000)
    e_ops = [qt.basis(dim, i).proj() for i in range(0, dim)]
    result = qt.mesolve(
        H,
        psi_in,
        t_list,
        e_ops=e_ops,
        options={
            "store_final_state": True,
            "normalize_output": normalize,
            "atol": 1e-30,
            "rtol": 1e-15,
        },
    )
    times = np.array(result.times)
    populations = [result.expect[i] for i in range(0, dim)]
    psi_out = result.final_state
    return times, populations, psi_out


# plot the populations of a subsystem as a function of time
def _plot_populations(times, populations, system):
    times_divided_by_pi = times / np.pi
    fig, ax = plt.subplots(layout="constrained")
    for i, pop in enumerate(populations):
        ax.plot(times_divided_by_pi, pop, label=r"$|$" + str(i) + r"$\rangle$")
    ax.set_title(r"System: " + str(system))
    ax.set_xlabel(r"$\Omega t/\pi$", fontsize=16)
    ax.set_ylabel("Populations", fontsize=16)
    ax.axhline(1, color="tomato", linestyle="--")
    ax.axhline(0, color="gray", linestyle="--")
    ax.legend(fontsize=12)
    plt.show()


# plot the pulse profile
def visualize_pulse(pulse, params, title_str=""):
    if isinstance(pulse, str):
        pulse = pulses_qutip.get_pulse(pulse)
    params = np.array(params)
    T = params[0]
    Delta_of_t, phi_of_t = pulse(params)
    t_list = np.linspace(0, T, 1000)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(t_list, Delta_of_t(t_list), label=r"Detuning $\Delta$")
    ax.plot(t_list, phi_of_t(t_list) / (2 * np.pi), label=r"Phase $\xi$")
    ax.set_xlabel(r"$\Omega t$", fontsize=16)
    ax.set_ylabel(r"$\Delta / \Omega \, , \; \xi/(2\pi)$", fontsize=16)
    # ax.set_xlim(-0.5, 8.0)
    # ax.set_ylim(-0.5, 0.5)
    ax.set_title(title_str, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid()
    plt.show()


# plot the population dynamics of all subsystems involved in a certain pulse
def visualize_subsystem_dynamics(
    n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params, decay, plot=True
):
    if isinstance(pulse, str):
        pulse = pulses_qutip.get_pulse(pulse)
    params = np.array(params)
    T = params[0]
    Delta_of_t, phi_of_t = pulse(params)
    Hs = _get_subsystem_Hs(n_atoms, Vnn, Vnnn, Delta_of_t, phi_of_t, decay)
    _, _, fidelity_fn = gates.get_subsystem_Hamiltonians(
        n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, decay
    )
    # time-evolve all subsystems and plot the populations
    output_states = []
    for H, dim in Hs:
        times, populations, psi_out = _time_evolution(H, dim, T, decay)
        if plot:
            _plot_populations(times, populations, H.__name__)
        output_states.append(psi_out)
    # calculate the fidelity from the different subsystems, as it is done during the training
    F = -1 * fidelity_fn(output_states)
    print("Qutip Fidelity (subsystems):   {f:.6f}".format(f=F))
    print("Qutip Infidelity (subsystems): {infi:.4e} \n".format(infi=1 - F))
    return F


if __name__ == "__main__":
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
    params = [
        10.97094681,
        0.19566367,
        0.43131090,
        -1.16460209,
        1.05669771,
        -0.70545851,
        0.88054914,
        -0.22756692,
    ]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # plot the pulse profile and the population dynamics of all subsystems involved
    visualize_pulse(pulse, params)
    visualize_subsystem_dynamics(
        n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, params, decay
    )
