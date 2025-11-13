import jax.numpy as jnp
from functools import partial


def H_2LS(Delta, Phi, Omega, decay, k):
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


def H_3LS(Delta, Phi, Omega, decay, V):
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


# Hamiltonian for subspace |111> -- (|r11>+|1r1>+|11r>) -- |r1r> -- (|11r>+|r11>-2|1r1>)  (Vnn = infinity)
def H_4LS(Delta, Phi, Omega, decay, Vnnn):
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi), 0.0, 0.0],
            [
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                Delta - 1j * 0.5 * decay,
                (1 / jnp.sqrt(3)) * Omega * jnp.exp(-1j * Phi),
                0.0,
            ],
            [
                0.0,
                (1 / jnp.sqrt(3)) * Omega * jnp.exp(1j * Phi),
                Vnnn + 2 * Delta - 1j * decay,
                (1 / jnp.sqrt(6)) * Omega * jnp.exp(1j * Phi),
            ],
            [
                0.0,
                0.0,
                (1 / jnp.sqrt(6)) * Omega * jnp.exp(-1j * Phi),
                Delta - 1j * 0.5 * decay,
            ],
        ]
    )


# Hamiltonian for subspace |111> -- (|r11>+|1r1>+|11r>) -- (|1rr>+|r1r>+|rr1>) -- |rrr>  (Vnn = Vnnn)
# TODO: rename or combine with other 4LS
def H_4LS_Vnnn(Delta, Phi, Omega, decay, Vnnn):
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
                Vnnn + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                3 * Vnnn + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


# Hamiltonian for subspace |1111> -- |W> -- |Z> -- |S> -- |rrr1>  (Vnn = infinity)
# |Z> = sqrt(3)/2 * |111r> - 1/sqrt(12) * (|11r1> + |1r11> + |r111>)
# |S> = 1/sqrt(3) * (|1rr1> + |r1r1> + |rr11>)
def H_5LS(Delta, Phi, Omega, decay, Vnnn):
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
                Vnnn + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Phi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Phi),
                3 * Vnnn + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


# Hamiltonian for subspace |111> -- |W> -- |X> -- |X'> -- |W'> -- |rrr>
def H_6LS(Delta, Phi, Omega, decay, Vnn, Vnnn):
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


# Hamiltonian for subspace |1111> -- |W> -- ... -- |rrrr>
def H_8LS(Delta, Phi, Omega, decay, Vnn, Vnnn):
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


H_2LS_1 = partial(H_2LS, k=1)
H_2LS_sqrt2 = partial(H_2LS, k=2)
H_2LS_sqrt3 = partial(H_2LS, k=3)
H_2LS_sqrt4 = partial(H_2LS, k=4)

H_3LS_Vnn = H_3LS
H_3LS_Vnnn = H_3LS
