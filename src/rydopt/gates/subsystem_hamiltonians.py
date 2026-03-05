import jax
import jax.numpy as jnp


def H_k_atoms_perfect_blockade(
    Delta_1: float, Delta_r: float, Xi: float, Omega: float, decay: float, k: int
) -> jax.Array:
    r""":math:`k` atoms, infinite Rydberg interaction between all atoms:

    .. image:: ../_static/k_atoms_perfect_blockade.png

    Args:
        Delta_1: Laser detuning of the qubit state |1>.
        Delta_r: Laser detuning of the Rydberg state |r>.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        k: Number of atoms.

    Returns:
        2-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            [-k * Delta_1, 0.5 * jnp.sqrt(k) * Omega * em],
            [
                0.5 * jnp.sqrt(k) * Omega * ep,
                -(k - 1) * Delta_1 - Delta_r - 1j * 0.5 * decay,
            ],
        ]
    )


def H_2_atoms(Delta_1: float, Delta_r: float, Xi: float, Omega: float, decay: float, V: float) -> jax.Array:
    r"""Two atoms, Rydberg interaction :math:`V` between atoms:

    .. image:: ../_static/2_atoms.png

    Args:
        Delta_1: Laser detuning of the qubit state |1>.
        Delta_r: Laser detuning of the Rydberg state |r>.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength.

    Returns:
        3-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            [-2 * Delta_1, 0.5 * jnp.sqrt(2) * Omega * em, 0],
            [
                0.5 * jnp.sqrt(2) * Omega * ep,
                -Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.5 * jnp.sqrt(2) * Omega * em,
            ],
            [
                0,
                0.5 * jnp.sqrt(2) * Omega * ep,
                V - 2 * Delta_r - 1j * decay,
            ],
        ]
    )


def H_3_atoms_inf_V(Delta_1: float, Delta_r: float, Xi: float, Omega: float, decay: float, V: float) -> jax.Array:
    r"""Three atoms arranged in an isosceles triangle,
    infinite Rydberg interaction between nearest neighbours, Rydberg interaction :math:`V` between next-nearest
    neighbours:

    .. image:: ../_static/3_atoms_inf_V.png

    Args:
        Delta_1: Laser detuning of the qubit state |1>.
        Delta_r: Laser detuning of the Rydberg state |r>.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength between next-nearest neighbours.

    Returns:
        4-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            [-3 * Delta_1, 0.5 * jnp.sqrt(3) * Omega * em, 0.0, 0.0],
            [
                0.5 * jnp.sqrt(3) * Omega * ep,
                -2 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.0,
                (1 / jnp.sqrt(3)) * Omega * em,
            ],
            [
                0.0,
                0.0,
                -2 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                (1 / jnp.sqrt(6)) * Omega * em,
            ],
            [
                0.0,
                (1 / jnp.sqrt(3)) * Omega * ep,
                (1 / jnp.sqrt(6)) * Omega * ep,
                V - Delta_1 - 2 * Delta_r - 1j * decay,
            ],
        ]
    )


def H_3_atoms_symmetric(Delta_1: float, Delta_r: float, Xi: float, Omega: float, decay: float, V: float) -> jax.Array:
    r"""Three atoms arranged in an equilateral triangle,
    Rydberg interaction :math:`V` between atoms:

    .. image:: ../_static/3_atoms_symmetric.png

    Args:
        Delta_1: Laser detuning of the qubit state |1>.
        Delta_r: Laser detuning of the Rydberg state |r>.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength.

    Returns:
        4-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            [-3 * Delta_1, 0.5 * jnp.sqrt(3) * Omega * em, 0.0, 0.0],
            [
                0.5 * jnp.sqrt(3) * Omega * ep,
                -2 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                Omega * em,
                0.0,
            ],
            [
                0.0,
                Omega * ep,
                V - Delta_1 - 2 * Delta_r - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * em,
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * ep,
                3 * V - 3 * Delta_r - 1j * 1.5 * decay,
            ],
        ]
    )


def H_3_atoms(
    Delta_1: float, Delta_r: float, Xi: float, Omega: float, decay: float, Vnn: float, Vnnn: float
) -> jax.Array:
    r"""Three atoms arranged in an isosceles triangle,
    Rydberg interaction :math:`V_{\mathrm{nn}}` between nearest neighbours, Rydberg interaction
    :math:`V_{\mathrm{nnn}}` between next-nearest neighbours:

    .. image:: ../_static/3_atoms.png

    Args:
        Delta_1: Laser detuning of the qubit state |1>.
        Delta_r: Laser detuning of the Rydberg state |r>.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        Vnn: Rydberg interaction strength between nearest neighbours.
        Vnnn: Rydberg interaction strength between next-nearest neighbours.

    Returns:
        6-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            [
                -3 * Delta_1,
                0.5 * jnp.sqrt(3) * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.5 * jnp.sqrt(3) * Omega * ep,
                -2 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.0,
                0.0,
                Omega * em,
                0.0,
            ],
            [
                0.0,
                0.0,
                -2 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.5 * Omega * em,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.5 * Omega * ep,
                (1 / 3) * Vnn + (2 / 3) * Vnnn - Delta_1 - 2 * Delta_r - 1j * decay,
                (1 / 3) * jnp.sqrt(2) * (Vnn - Vnnn),
                0.0,
            ],
            [
                0.0,
                Omega * ep,
                0.0,
                (1 / 3) * jnp.sqrt(2) * (Vnn - Vnnn),
                (2 / 3) * Vnn + (1 / 3) * Vnnn - Delta_1 - 2 * Delta_r - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * em,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * ep,
                2 * Vnn + Vnnn - 3 * Delta_r - 1j * 1.5 * decay,
            ],
        ]
    )


def H_4_atoms_inf_V(Delta_1: float, Delta_r: float, Xi: float, Omega: float, decay: float, V: float) -> jax.Array:
    r"""Four atoms arranged in a pyramid,
    infinite Rydberg interaction between nearest neighbours, Rydberg interaction :math:`V` between
    next-nearest neighbours:

    .. image:: ../_static/4_atoms_inf_V.png

    Args:
        Delta_1: Laser detuning of the qubit state |1>.
        Delta_r: Laser detuning of the Rydberg state |r>.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength between next-nearest neighbours.

    Returns:
        5-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            [-4 * Delta_1, Omega * em, 0.0, 0.0, 0.0],
            [
                Omega * ep,
                -3 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * em,
                0.0,
            ],
            [
                0.0,
                0.0,
                -3 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.5 * Omega * em,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(3) * Omega * ep,
                0.5 * Omega * ep,
                V - 2 * Delta_1 - 2 * Delta_r - 1j * decay,
                0.5 * jnp.sqrt(3) * Omega * em,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(3) * Omega * ep,
                3 * V - Delta_1 - 3 * Delta_r - 1j * 1.5 * decay,
            ],
        ]
    )


def H_4_atoms_symmetric(Delta_1: float, Delta_r: float, Xi: float, Omega: float, decay: float, V: float) -> jax.Array:
    r"""Four atoms arranged in a tetrahedron,
    Rydberg interaction :math:`V` between atoms:

    .. image:: ../_static/4_atoms_symmetric.png

    Args:
        Delta_1: Laser detuning of the qubit state |1>.
        Delta_r: Laser detuning of the Rydberg state |r>.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        V: Rydberg interaction strength.

    Returns:
        5-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            [-4 * Delta_1, Omega * em, 0.0, 0.0, 0.0],
            [
                Omega * ep,
                -3 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.5 * jnp.sqrt(6) * Omega * em,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(6) * Omega * ep,
                V - 2 * Delta_1 - 2 * Delta_r - 1j * decay,
                0.5 * jnp.sqrt(6) * Omega * em,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * ep,
                3 * V - Delta_1 - 3 * Delta_r - 1j * 1.5 * decay,
                Omega * em,
            ],
            [
                0.0,
                0.0,
                0.0,
                Omega * ep,
                6 * V - 4 * Delta_r - 1j * 2 * decay,
            ],
        ]
    )


def H_4_atoms(
    Delta_1: float, Delta_r: float, Xi: float, Omega: float, decay: float, Vnn: float, Vnnn: float
) -> jax.Array:
    r"""Four atoms arranged in a pyramid,
    Rydberg interaction :math:`V_{\mathrm{nn}}` between nearest neighbours, Rydberg interaction
    :math:`V_{\mathrm{nnn}}` between next-nearest neighbours:

    .. image:: ../_static/4_atoms.png

    Args:
        Delta_1: Laser detuning of the qubit state |1>.
        Delta_r: Laser detuning of the Rydberg state |r>.
        Xi: Laser phase.
        Omega: Rabi frequency amplitude.
        decay: Rydberg-decay rate.
        Vnn: Rydberg interaction strength between nearest neighbours.
        Vnnn: Rydberg interaction strength between next-nearest neighbours.

    Returns:
        8-level system Hamiltonian.

    """
    em = jnp.exp(-1j * Xi)
    ep = jnp.exp(1j * Xi)

    return jnp.array(
        [
            [-4 * Delta_1, Omega * em, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                Omega * ep,
                -3 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * em,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                -3 * Delta_1 - Delta_r - 1j * 0.5 * decay,
                0.0,
                0.5 * jnp.sqrt(2) * Omega * em,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5 * jnp.sqrt(6) * Omega * ep,
                0.0,
                0.5 * Vnn + 0.5 * Vnnn - 2 * Delta_1 - 2 * Delta_r - 1j * decay,
                0.5 * (Vnn - Vnnn),
                0.0,
                0.5 * jnp.sqrt(6) * Omega * em,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.5 * jnp.sqrt(2) * Omega * ep,
                0.5 * (Vnn - Vnnn),
                0.5 * Vnn + 0.5 * Vnnn - 2 * Delta_1 - 2 * Delta_r - 1j * decay,
                0.5 * jnp.sqrt(2) * Omega * em,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(2) * Omega * ep,
                0.5 * Vnn + 2.5 * Vnnn - Delta_1 - 3 * Delta_r - 1j * 1.5 * decay,
                0.5 * jnp.sqrt(3) * (Vnn - Vnnn),
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.5 * jnp.sqrt(6) * Omega * ep,
                0.0,
                0.5 * jnp.sqrt(3) * (Vnn - Vnnn),
                1.5 * Vnn + 1.5 * Vnnn - Delta_1 - 3 * Delta_r - 1j * 1.5 * decay,
                Omega * em,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                Omega * ep,
                3 * Vnn + 3 * Vnnn - 4 * Delta_r - 1j * 2 * decay,
            ],
        ]
    )
