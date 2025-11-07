import jax.numpy as jnp
from rydopt import pulses, pulse_visualization, pulse_verification, train_single_gate


# run pulse optimizations locally

if __name__ == "__main__":
    jnp.set_printoptions(linewidth=400)

    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 3

    # Rydberg interaction strengths (use float("inf") for perfect blockade)
    Vnn = float("inf")  # float("inf")
    Vnnn = float("inf")
    decay = 0.000

    # target gate phases
    theta = jnp.pi  # set theta=None, eps=None if both are arbitrary / set theta='eps', eps=None if both are arbitrary but must be equal
    eps = jnp.pi  # set eps=None if eps is arbitrary
    lamb = jnp.pi
    delta = 0  # set delta=None if delta is arbitrary
    kappa = 0

    # pulse type
    pulse = pulses.pulse_phase_sin_crab

    # initial parameters
    initial_params = jnp.array([11.0, 0.2, 0.4, -1.2, 1.1, -0.7, 0.9, -0.2])

    # gate search parameters
    T_default = 11.0
    N_searches = 20
    N_params = 8

    # optimization settings
    N_epochs = 1000
    learning_rate = 0.05
    T_penalty = 0.0

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # uncomment one of the following two lines
    params = train_single_gate(
        n_atoms,
        Vnn,
        Vnnn,
        theta,
        eps,
        lamb,
        delta,
        kappa,
        pulse,
        initial_params,
        N_epochs,
        learning_rate,
        T_penalty,
        decay,
    )
    # params = gate_search(n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, pulse, T_default, N_searches, N_params, N_epochs, learning_rate, T_penalty, decay)

    # visualize the pulse and the dynamics in the subsystems (using qutip)
    pulse_visualization.visualize_pulse(pulse.__name__, params)
    pulse_visualization.visualize_subsystem_dynamics(
        n_atoms,
        Vnn,
        Vnnn,
        theta,
        eps,
        lamb,
        delta,
        kappa,
        pulse.__name__,
        params,
        decay,
    )

    # verify the pulse using the full Hamiltonian (using qutip)
    pulse_verification.verify(
        n_atoms,
        Vnn,
        Vnnn,
        theta,
        eps,
        lamb,
        delta,
        kappa,
        pulse.__name__,
        params,
        decay,
    )
