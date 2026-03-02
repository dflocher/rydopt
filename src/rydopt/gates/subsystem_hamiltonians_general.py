import jax
import jax.numpy as jnp


def H_1_atom_general(
    Delta: float,
    Xi: float,
    Omega: float,
    decay: float,
    s1: float = 1.0,
) -> jax.Array:
    r"""One atom with arbitrary scaling of the Rabi frequency.

    Basis ordering: :math:`|0\rangle, |1\rangle`.

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        s1: Rabi frequency scaling factor for the atom.

    Returns:
        2-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            # |0>
            [0.0, 0.5 * s1 * Omega * em],
            # |1>
            [0.5 * s1 * Omega * ep, Delta - 1j * 0.5 * decay],
        ]
    )


def H_2_atoms_general(
    Delta: float,
    Xi: float,
    Omega: float,
    decay: float,
    V12: float,
    s1: float = 1.0,
    s2: float = 1.0,
) -> jax.Array:
    r"""Two atoms with arbitrary scaling of Rabi frequencies and arbitrary Rydberg interaction.

    Basis ordering: :math:`|00\rangle, |01\rangle, |10\rangle, |11\rangle`.

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V12: Rydberg interaction strength between atoms 1 and 2.
        s1: Rabi frequency scaling factor for atom 1.
        s2: Rabi frequency scaling factor for atom 2.

    Returns:
        4-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            # |00>
            [
                0.0,
                0.5 * s2 * Omega * em,
                0.5 * s1 * Omega * em,
                0.0,
            ],
            # |01>
            [
                0.5 * s2 * Omega * ep,
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * s1 * Omega * em,
            ],
            # |10>
            [
                0.5 * s1 * Omega * ep,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s2 * Omega * em,
            ],
            # |11>
            [
                0.0,
                0.5 * s1 * Omega * ep,
                0.5 * s2 * Omega * ep,
                V12 + 2 * Delta - 1j * decay,
            ],
        ]
    )


def H_3_atoms_general(
    Delta: float,
    Xi: float,
    Omega: float,
    decay: float,
    V12: float,
    V13: float,
    V23: float,
    s1: float = 1.0,
    s2: float = 1.0,
    s3: float = 1.0,
) -> jax.Array:
    r"""Three atoms with arbitrary scaling of Rabi frequencies and arbitrary Rydberg interactions.

    Basis ordering: :math:`|000\rangle, |001\rangle, |010\rangle, |011\rangle,
    |100\rangle, |101\rangle, |110\rangle, |111\rangle`.

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V12: Rydberg interaction strength between atoms 1 and 2.
        V13: Rydberg interaction strength between atoms 1 and 3.
        V23: Rydberg interaction strength between atoms 2 and 3.
        s1: Rabi frequency scaling factor for atom 1.
        s2: Rabi frequency scaling factor for atom 2.
        s3: Rabi frequency scaling factor for atom 3.

    Returns:
        8-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            # |000>
            [
                0.0,
                0.5 * s3 * Omega * em,
                0.5 * s2 * Omega * em,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
                0.0,
                0.0,
            ],
            # |001>
            [
                0.5 * s3 * Omega * ep,
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * s2 * Omega * em,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
                0.0,
            ],
            # |010>
            [
                0.5 * s2 * Omega * ep,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s3 * Omega * em,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
            ],
            # |011>
            [
                0.0,
                0.5 * s2 * Omega * ep,
                0.5 * s3 * Omega * ep,
                V23 + 2 * Delta - 1j * decay,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
            ],
            # |100>
            [
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s3 * Omega * em,
                0.5 * s2 * Omega * em,
                0.0,
            ],
            # |101>
            [
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.5 * s3 * Omega * ep,
                V13 + 2 * Delta - 1j * decay,
                0.0,
                0.5 * s2 * Omega * em,
            ],
            # |110>
            [
                0.0,
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.5 * s2 * Omega * ep,
                0.0,
                V12 + 2 * Delta - 1j * decay,
                0.5 * s3 * Omega * em,
            ],
            # |111>
            [
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.5 * s2 * Omega * ep,
                0.5 * s3 * Omega * ep,
                V12 + V23 + V13 + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


def H_4_atoms_general(
    Delta: float,
    Xi: float,
    Omega: float,
    decay: float,
    V12: float,
    V13: float,
    V14: float,
    V23: float,
    V24: float,
    V34: float,
    s1: float = 1.0,
    s2: float = 1.0,
    s3: float = 1.0,
    s4: float = 1.0,
) -> jax.Array:
    r"""Four atoms with arbitrary scaling of Rabi frequencies and arbitrary Rydberg interactions.

    Basis ordering: :math:`|0000\rangle, |0001\rangle, |0010\rangle, |0011\rangle, |0100\rangle, \ldots, |1111\rangle`.

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V12: Rydberg interaction strength between atoms 1 and 2.
        V13: Rydberg interaction strength between atoms 1 and 3.
        V14: Rydberg interaction strength between atoms 1 and 4.
        V23: Rydberg interaction strength between atoms 2 and 3.
        V24: Rydberg interaction strength between atoms 2 and 4.
        V34: Rydberg interaction strength between atoms 3 and 4.
        s1: Rabi frequency scaling factor for atom 1.
        s2: Rabi frequency scaling factor for atom 2.
        s3: Rabi frequency scaling factor for atom 3.
        s4: Rabi frequency scaling factor for atom 4.

    Returns:
        16-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            # |0000>
            [
                0.0,
                0.5 * s4 * Omega * em,
                0.5 * s3 * Omega * em,
                0.0,
                0.5 * s2 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            # |0001>
            [
                0.5 * s4 * Omega * ep,
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * s3 * Omega * em,
                0.0,
                0.5 * s2 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            # |0010>
            [
                0.5 * s3 * Omega * ep,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s4 * Omega * em,
                0.0,
                0.0,
                0.5 * s2 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            # |0011>
            [
                0.0,
                0.5 * s3 * Omega * ep,
                0.5 * s4 * Omega * ep,
                V34 + 2 * Delta - 1j * decay,
                0.0,
                0.0,
                0.0,
                0.5 * s2 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            # |0100>
            [
                0.5 * s2 * Omega * ep,
                0.0,
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s4 * Omega * em,
                0.5 * s3 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
                0.0,
                0.0,
            ],
            # |0101>
            [
                0.0,
                0.5 * s2 * Omega * ep,
                0.0,
                0.0,
                0.5 * s4 * Omega * ep,
                V24 + 2 * Delta - 1j * decay,
                0.0,
                0.5 * s3 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
                0.0,
            ],
            # |0110>
            [
                0.0,
                0.0,
                0.5 * s2 * Omega * ep,
                0.0,
                0.5 * s3 * Omega * ep,
                0.0,
                V23 + 2 * Delta - 1j * decay,
                0.5 * s4 * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
                0.0,
            ],
            # |0111>
            [
                0.0,
                0.0,
                0.0,
                0.5 * s2 * Omega * ep,
                0.0,
                0.5 * s3 * Omega * ep,
                0.5 * s4 * Omega * ep,
                V23 + V24 + V34 + 3 * Delta - 1j * 1.5 * decay,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * em,
            ],
            # |1000>
            [
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s4 * Omega * em,
                0.5 * s3 * Omega * em,
                0.0,
                0.5 * s2 * Omega * em,
                0.0,
                0.0,
                0.0,
            ],
            # |1001>
            [
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s4 * Omega * ep,
                V14 + 2 * Delta - 1j * decay,
                0.0,
                0.5 * s3 * Omega * em,
                0.0,
                0.5 * s2 * Omega * em,
                0.0,
                0.0,
            ],
            # |1010>
            [
                0.0,
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s3 * Omega * ep,
                0.0,
                V13 + 2 * Delta - 1j * decay,
                0.5 * s4 * Omega * em,
                0.0,
                0.0,
                0.5 * s2 * Omega * em,
                0.0,
            ],
            # |1011>
            [
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s3 * Omega * ep,
                0.5 * s4 * Omega * ep,
                V13 + V14 + V34 + 3 * Delta - 1j * 1.5 * decay,
                0.0,
                0.0,
                0.0,
                0.5 * s2 * Omega * em,
            ],
            # |1100>
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                0.5 * s2 * Omega * ep,
                0.0,
                0.0,
                0.0,
                V12 + 2 * Delta - 1j * decay,
                0.5 * s4 * Omega * em,
                0.5 * s3 * Omega * em,
                0.0,
            ],
            # |1101>
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                0.5 * s2 * Omega * ep,
                0.0,
                0.0,
                0.5 * s4 * Omega * ep,
                V12 + V14 + V24 + 3 * Delta - 1j * 1.5 * decay,
                0.0,
                0.5 * s3 * Omega * em,
            ],
            # |1110>
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                0.5 * s2 * Omega * ep,
                0.0,
                0.5 * s3 * Omega * ep,
                0.0,
                V12 + V13 + V23 + 3 * Delta - 1j * 1.5 * decay,
                0.5 * s4 * Omega * em,
            ],
            # |1111>
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * s1 * Omega * ep,
                0.0,
                0.0,
                0.0,
                0.5 * s2 * Omega * ep,
                0.0,
                0.5 * s3 * Omega * ep,
                0.5 * s4 * Omega * ep,
                V12 + V13 + V14 + V23 + V24 + V34 + 4 * Delta - 1j * 2 * decay,
            ],
        ]
    )
