import jax.numpy as jnp


def H_k_atoms_perfect_blockade(Delta, Phi, Omega, decay, k):
    # k=1: Hamiltonian for subspace |1> -- |r>
    # k=2: Hamiltonian for subspace |11> -- |W2> = |1r> + |r1>  (Vnn = infinity)
    # k=3: Hamiltonian for subspace |111> -- |W3>  (Vnn = Vnnn = infinity)
    # k=4: Hamiltonian for subspace |1111> -- |W4>  (Vnn = Vnnn = Vnnnn = infinity)
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(k) * Omega * jnp.exp(-1j * Phi)],
            [0.5 * jnp.sqrt(k) * Omega * jnp.exp(1j * Phi), Delta - 1j * 0.5 * decay],
        ]
    )


def H_2_atoms(Delta, Phi, Omega, decay, V):
    # V=Vnn: Hamiltonian for subspace |011> -- (|01r> + |0r1>) -- |0rr>
    # V=Vnnn: Hamiltonian for subspace |101> -- (|10r> + |r01>) -- |r0r>
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(2) * Omega * jnp.exp(-1j * Phi), 0],
            [
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(1j * Phi),
                2 * Delta + V - 1j * decay,
            ],
        ]
    )


# Hamiltonian for subspace |111> -- (|r11>+|1r1>+|11r>) -- (|11r>+|r11>-2|1r1>) -- |r1r>  (Vnn = infinity)
def H_3_atoms_inf_V(Delta, Phi, Omega, decay, V):
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi), 0.0, 0.0],
            [
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                0.0,
                (1 / jnp.sqrt(3)) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                (1 / jnp.sqrt(6)) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                (1 / jnp.sqrt(3)) * Omega * jnp.exp(1j * Phi),
                (1 / jnp.sqrt(6)) * Omega * jnp.exp(1j * Phi),
                V + 2 * Delta - 1j * decay,
            ],
        ]
    )


# Hamiltonian for subspace |111> -- (|r11>+|1r1>+|11r>) -- (|1rr>+|r1r>+|rr1>) -- |rrr>  (Vnn = Vnnn)
def H_3_atoms_symmetric(Delta, Phi, Omega, decay, V):
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi), 0.0, 0.0],
            [
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                Omega * jnp.exp(-1j * Phi),
                0.0,
            ],
            [
                0.0,
                Omega * jnp.exp(1j * Phi),
                V + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                3 * V + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


# Hamiltonian for subspace |111> -- |W> -- |X> -- |X'> -- |W'> -- |rrr>
def H_3_atoms(Delta, Phi, Omega, decay, Vnn, Vnnn):
    return jnp.array(
        [
            [
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.0,
                Omega * jnp.exp(-1j * Phi),
                0.0,
            ],
            [
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                -0.5 * Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                -0.5 * Omega * jnp.exp(1j * Phi),
                (1 / 3) * Vnn + (2 / 3) * Vnnn + 2 * Delta - 1j * decay,
                (1 / 3) * jnp.sqrt(2) * (Vnn - Vnnn),
                0.0,
            ],
            [
                0.0,
                Omega * jnp.exp(1j * Phi),
                0.0,
                (1 / 3) * jnp.sqrt(2) * (Vnn - Vnnn),
                (2 / 3) * Vnn + (1 / 3) * Vnnn + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                2 * Vnn + Vnnn + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


# Hamiltonian for subspace |1111> -- |W> -- |Z> -- |S> -- |rrr1>  (Vnn = infinity)
# |Z> = sqrt(3)/2 * |111r> - 1/sqrt(12) * (|11r1> + |1r11> + |r111>)
# |S> = 1/sqrt(3) * (|1rr1> + |r1r1> + |rr11>)
def H_4_atoms_inf_V(Delta, Phi, Omega, decay, V):
    return jnp.array(
        [
            [0.0, Omega * jnp.exp(-1j * Phi), 0.0, 0.0, 0.0],
            [
                Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
                0.0,
            ],
            [
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * Omega * jnp.exp(-1j * Phi),
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                0.5 * Omega * jnp.exp(1j * Phi),
                V + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                3 * V + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


# Hamiltonian for subspace |1111> -- |W> -- ... -- |rrrr>  (Vnn = Vnnn)
def H_4_atoms_symmetric(Delta, Phi, Omega, decay, V):
    return jnp.array(
        [
            [0.0, Omega * jnp.exp(-1j * Phi), 0.0, 0.0, 0.0],
            [
                Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(1j * Phi),
                V + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(-1j * Phi),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(1j * Phi),
                3 * V + 3 * Delta - 1j * 1.5 * decay,
                Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.0,
                Omega * jnp.exp(1j * Phi),
                6 * V + 4 * Delta - 1j * 2 * decay,
            ],
        ]
    )


# Hamiltonian for subspace |1111> -- |W> -- ... -- |rrrr>
def H_4_atoms(Delta, Phi, Omega, decay, Vnn, Vnnn):
    return jnp.array(
        [
            [0.0, Omega * jnp.exp(-1j * Phi), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                -0.5 * Omega * jnp.exp(-1j * Phi),
                0.5 * Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                -0.5 * Omega * jnp.exp(1j * Phi),
                2 * Delta - 1j * decay + Vnn,
                0.0,
                Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                0.5 * Omega * jnp.exp(1j * Phi),
                0.0,
                2 * Delta - 1j * decay + Vnnn,
                0.5 * Omega * jnp.exp(-1j * Phi),
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                Omega * jnp.exp(1j * Phi),
                0.5 * Omega * jnp.exp(1j * Phi),
                3 * Delta - 1j * 1.5 * decay + 2 * Vnn + Vnnn,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                0.0,
                3 * Delta - 1j * 1.5 * decay + 3 * Vnnn,
                0.5 * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                0.5 * Omega * jnp.exp(1j * Phi),
                4 * Delta - 1j * 2 * decay + 3 * Vnn + 3 * Vnnn,
            ],
        ]
    )


# Hamiltonian for subspace |1111> -- |W> -- ... -- |rrrr>
# TODO: compare both 4_atom Hamiltonians
def H_4_atoms_v2(Delta, Phi, Omega, decay, Vnn, Vnnn):
    return jnp.array(
        [
            [0.0, Omega * jnp.exp(-1j * Phi), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(1j * Phi),
                0.0,
                0.5 * Vnn + 0.5 * Vnnn + 2 * Delta - 1j * decay,
                0.5 * (Vnnn - Vnn),
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(-1j * Phi),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(1j * Phi),
                0.5 * (Vnnn - Vnn),
                0.5 * Vnn + 0.5 * Vnnn + 2 * Delta - 1j * decay,
                -0.5 * jnp.sqrt(2) * Omega * jnp.exp(-1j * Phi),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -0.5 * jnp.sqrt(2) * Omega * jnp.exp(1j * Phi),
                0.5 * Vnn + 2.5 * Vnnn + 3 * Delta - 1j * 1.5 * decay,
                0.5 * jnp.sqrt(3) * (Vnn - Vnnn),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(1j * Phi),
                0.0,
                0.5 * jnp.sqrt(3) * (Vnn - Vnnn),
                1.5 * Vnn + 1.5 * Vnnn + 3 * Delta - 1j * 1.5 * decay,
                Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                Omega * jnp.exp(1j * Phi),
                3 * Vnn + 3 * Vnnn + 4 * Delta - 1j * 2 * decay,
            ],
        ]
    )
