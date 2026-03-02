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
    return jnp.array(
        [
            # |0>
            [0.0, 0.5 * s1 * Omega * jnp.exp(-1j * Xi)],
            # |1>
            [0.5 * s1 * Omega * jnp.exp(1j * Xi), Delta - 1j * 0.5 * decay],
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

    Basis ordering: :math:`|00\rangle, |10\rangle, |01\rangle, |11\rangle`.

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
    return jnp.array(
        [
            # |00>
            [
                0.0,
                0.5 * s1 * Omega * jnp.exp(-1j * Xi),
                0.5 * s2 * Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            # |10>
            [
                0.5 * s1 * Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * s2 * Omega * jnp.exp(-1j * Xi),
            ],
            # |01>
            [
                0.5 * s2 * Omega * jnp.exp(1j * Xi),
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s1 * Omega * jnp.exp(-1j * Xi),
            ],
            # |11>
            [
                0.0,
                0.5 * s2 * Omega * jnp.exp(1j * Xi),
                0.5 * s1 * Omega * jnp.exp(1j * Xi),
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

    Basis ordering: :math:`|000\rangle, |100\rangle, |010\rangle, |110\rangle,
    |001\rangle, |101\rangle, |011\rangle, |111\rangle`.

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
    return jnp.array(
        [
            # |000>
            [
                0.0,
                0.5 * s1 * Omega * jnp.exp(-1j * Xi),
                0.5 * s2 * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.5 * s3 * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.0,
                0.0,
            ],
            # |100>
            [
                0.5 * s1 * Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * s2 * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.5 * s3 * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.0,
            ],
            # |010>
            [
                0.5 * s2 * Omega * jnp.exp(1j * Xi),
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s1 * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.0,
                0.5 * s3 * Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            # |110>
            [
                0.0,
                0.5 * s2 * Omega * jnp.exp(1j * Xi),
                0.5 * s1 * Omega * jnp.exp(1j * Xi),
                V23 + 2 * Delta - 1j * decay,
                0.0,
                0.0,
                0.0,
                0.5 * s3 * Omega * jnp.exp(-1j * Xi),
            ],
            # |001>
            [
                0.5 * s3 * Omega * jnp.exp(1j * Xi),
                0.0,
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * s1 * Omega * jnp.exp(-1j * Xi),
                0.5 * s2 * Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            # |101>
            [
                0.0,
                0.5 * s3 * Omega * jnp.exp(1j * Xi),
                0.0,
                0.0,
                0.5 * s1 * Omega * jnp.exp(1j * Xi),
                V13 + 2 * Delta - 1j * decay,
                0.0,
                0.5 * s2 * Omega * jnp.exp(-1j * Xi),
            ],
            # |011>
            [
                0.0,
                0.0,
                0.5 * s3 * Omega * jnp.exp(1j * Xi),
                0.0,
                0.5 * s2 * Omega * jnp.exp(1j * Xi),
                0.0,
                V12 + 2 * Delta - 1j * decay,
                0.5 * s1 * Omega * jnp.exp(-1j * Xi),
            ],
            # |111>
            [
                0.0,
                0.0,
                0.0,
                0.5 * s3 * Omega * jnp.exp(1j * Xi),
                0.0,
                0.5 * s2 * Omega * jnp.exp(1j * Xi),
                0.5 * s1 * Omega * jnp.exp(1j * Xi),
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

    Basis ordering: :math:`|0000\rangle, |1000\rangle, |0100\rangle, |1100\rangle, |0010\rangle, \ldots, |1111\rangle`.

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
    s = (s1, s2, s3, s4)
    V = {(1, 2): V12, (1, 3): V13, (1, 4): V14, (2, 3): V23, (2, 4): V24, (3, 4): V34}

    H = [[0.0 for _ in range(16)] for _ in range(16)]

    for i in range(16):
        # Bits a1..a4 (a1 is LSB / first ket digit)
        a1 = (i >> 0) & 1
        a2 = (i >> 1) & 1
        a3 = (i >> 2) & 1
        a4 = (i >> 3) & 1
        bits = (a1, a2, a3, a4)

        n_exc = a1 + a2 + a3 + a4
        diag = n_exc * Delta - 1j * 0.5 * n_exc * decay

        # Add pairwise interactions for simultaneously excited atoms
        if a1 and a2:
            diag = diag + V[(1, 2)]
        if a1 and a3:
            diag = diag + V[(1, 3)]
        if a1 and a4:
            diag = diag + V[(1, 4)]
        if a2 and a3:
            diag = diag + V[(2, 3)]
        if a2 and a4:
            diag = diag + V[(2, 4)]
        if a3 and a4:
            diag = diag + V[(3, 4)]

        H[i][i] = diag

        # Laser couplings
        for k in range(4):  # k=0..3 corresponds to atom 1..4
            if bits[k] == 0:
                j = i | (1 << k)  # excite atom k+1
                amp = 0.5 * s[k] * Omega
                H[i][j] = amp * em
                H[j][i] = amp * ep

    return jnp.array(H)
