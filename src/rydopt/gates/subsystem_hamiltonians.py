import jax.numpy as jnp


def H_k_atoms_perfect_blockade(Delta: float, Xi: float, Omega: float, decay: float, k: int) -> jnp.ndarray:
    r""":math:`k` atoms, infinite Rydberg interaction between all atoms:

    .. image:: ../_static/k_atoms_perfect_blockade.png

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        k: Number of atoms.

    Returns:
        2-level system Hamiltonian.

    """
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(k) * Omega * jnp.exp(-1j * Xi)],
            [0.5 * jnp.sqrt(k) * Omega * jnp.exp(1j * Xi), Delta - 1j * 0.5 * decay],
        ]
    )


def H_2_atoms(Delta: float, Xi: float, Omega: float, decay: float, V: float) -> jnp.ndarray:
    r"""Two atoms, Rydberg interaction :math:`V` between atoms:

    .. image:: ../_static/2_atoms.png

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength.

    Returns:
        3-level system Hamiltonian.

    """
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(2) * Omega * jnp.exp(-1j * Xi), 0],
            [
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(-1j * Xi),
            ],
            [
                0,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(1j * Xi),
                2 * Delta + V - 1j * decay,
            ],
        ]
    )


def H_3_atoms_inf_V(Delta: float, Xi: float, Omega: float, decay: float, V: float) -> jnp.ndarray:
    r"""Three atoms arranged in an isosceles triangle,
    infinite Rydberg interaction between nearest neighbours, Rydberg interaction :math:`V` between next-nearest
    neighbours:

    .. image:: ../_static/3_atoms_inf_V.png

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength between next-nearest neighbours.

    Returns:
        4-level system Hamiltonian.

    """
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Xi), 0.0, 0.0],
            [
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                0.0,
                (1 / jnp.sqrt(3)) * Omega * jnp.exp(-1j * Xi),
            ],
            [
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                (1 / jnp.sqrt(6)) * Omega * jnp.exp(-1j * Xi),
            ],
            [
                0.0,
                (1 / jnp.sqrt(3)) * Omega * jnp.exp(1j * Xi),
                (1 / jnp.sqrt(6)) * Omega * jnp.exp(1j * Xi),
                V + 2 * Delta - 1j * decay,
            ],
        ]
    )


def H_3_atoms_symmetric(Delta: float, Xi: float, Omega: float, decay: float, V: float) -> jnp.ndarray:
    r"""Three atoms arranged in an equilateral triangle,
    Rydberg interaction :math:`V` between atoms:

    .. image:: ../_static/3_atoms_symmetric.png

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength.

    Returns:
        4-level system Hamiltonian.

    """
    return jnp.array(
        [
            [0.0, 0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Xi), 0.0, 0.0],
            [
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            [
                0.0,
                Omega * jnp.exp(1j * Xi),
                V + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Xi),
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Xi),
                3 * V + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


def H_3_atoms(Delta: float, Xi: float, Omega: float, decay: float, Vnn: float, Vnnn: float) -> jnp.ndarray:
    r"""Three atoms arranged in an isosceles triangle,
    Rydberg interaction :math:`V_{\mathrm{nn}}` between nearest neighbours, Rydberg interaction
    :math:`V_{\mathrm{nnn}}` between next-nearest neighbours:

    .. image:: ../_static/3_atoms.png

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        Vnn: Rydberg interaction strength between nearest neighbours.
        Vnnn: Rydberg interaction strength between next-nearest neighbours.

    Returns:
        6-level system Hamiltonian.

    """
    return jnp.array(
        [
            [
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.0,
                Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            [
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.5 * Omega * jnp.exp(1j * Xi),
                (1 / 3) * Vnn + (2 / 3) * Vnnn + 2 * Delta - 1j * decay,
                (1 / 3) * jnp.sqrt(2) * (Vnn - Vnnn),
                0.0,
            ],
            [
                0.0,
                Omega * jnp.exp(1j * Xi),
                0.0,
                (1 / 3) * jnp.sqrt(2) * (Vnn - Vnnn),
                (2 / 3) * Vnn + (1 / 3) * Vnnn + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Xi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Xi),
                2 * Vnn + Vnnn + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


def H_4_atoms_inf_V(Delta: float, Xi: float, Omega: float, decay: float, V: float) -> jnp.ndarray:
    r"""Four atoms arranged in a pyramid,
    infinite Rydberg interaction between nearest neighbours, Rydberg interaction :math:`V` between
    next-nearest neighbours:

    .. image:: ../_static/4_atoms_inf_V.png

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength between next-nearest neighbours.

    Returns:
        5-level system Hamiltonian.

    """
    return jnp.array(
        [
            [0.0, Omega * jnp.exp(-1j * Xi), 0.0, 0.0, 0.0],
            [
                Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            [
                0.0,
                0.0,
                Delta - 1j * 0.5 * decay,
                0.5 * Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Xi),
                0.5 * Omega * jnp.exp(1j * Xi),
                V + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(-1j * Xi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * jnp.exp(1j * Xi),
                3 * V + 3 * Delta - 1j * 1.5 * decay,
            ],
        ]
    )


def H_4_atoms_symmetric(Delta: float, Xi: float, Omega: float, decay: float, V: float) -> jnp.ndarray:
    r"""Four atoms arranged in a tetrahedron,
    Rydberg interaction :math:`V` between atoms:

    .. image:: ../_static/4_atoms_symmetric.png

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength.

    Returns:
        5-level system Hamiltonian.

    """
    return jnp.array(
        [
            [0.0, Omega * jnp.exp(-1j * Xi), 0.0, 0.0, 0.0],
            [
                Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(1j * Xi),
                V + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(1j * Xi),
                3 * V + 3 * Delta - 1j * 1.5 * decay,
                Omega * jnp.exp(-1j * Xi),
            ],
            [
                0.0,
                0.0,
                0.0,
                Omega * jnp.exp(1j * Xi),
                6 * V + 4 * Delta - 1j * 2 * decay,
            ],
        ]
    )


def H_4_atoms(Delta: float, Xi: float, Omega: float, decay: float, Vnn: float, Vnnn: float) -> jnp.ndarray:
    r"""Four atoms arranged in a pyramid,
    Rydberg interaction :math:`V_{\mathrm{nn}}` between nearest neighbours, Rydberg interaction
    :math:`V_{\mathrm{nnn}}` between next-nearest neighbours:

    .. image:: ../_static/4_atoms.png

    Args:
        Delta: Laser detuning.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        Vnn: Rydberg interaction strength between nearest neighbours.
        Vnnn: Rydberg interaction strength between next-nearest neighbours.

    Returns:
        8-level system Hamiltonian.

    """
    return jnp.array(
        [
            [0.0, Omega * jnp.exp(-1j * Xi), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                Omega * jnp.exp(1j * Xi),
                Delta - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(-1j * Xi),
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
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(1j * Xi),
                0.0,
                0.5 * Vnn + 0.5 * Vnnn + 2 * Delta - 1j * decay,
                0.5 * (Vnn - Vnnn),
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(-1j * Xi),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(1j * Xi),
                0.5 * (Vnn - Vnnn),
                0.5 * Vnn + 0.5 * Vnnn + 2 * Delta - 1j * decay,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(-1j * Xi),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(2) * Omega * jnp.exp(1j * Xi),
                0.5 * Vnn + 2.5 * Vnnn + 3 * Delta - 1j * 1.5 * decay,
                0.5 * jnp.sqrt(3) * (Vnn - Vnnn),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * jnp.exp(1j * Xi),
                0.0,
                0.5 * jnp.sqrt(3) * (Vnn - Vnnn),
                1.5 * Vnn + 1.5 * Vnnn + 3 * Delta - 1j * 1.5 * decay,
                Omega * jnp.exp(-1j * Xi),
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                Omega * jnp.exp(1j * Xi),
                3 * Vnn + 3 * Vnnn + 4 * Delta - 1j * 2 * decay,
            ],
        ]
    )
