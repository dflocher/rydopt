from functools import partial
import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_k_atoms_perfect_blockade,
    H_2_atoms,
)
from math import isinf


class TwoQubitGate(Gate):
    r"""Class that describes a gate on two atoms.
    The physical setting is described by the interaction strength between the atoms, :math:`V_{\mathrm{nn}}`,
    and the decay strength from Rydberg states, :math:`\gamma`.
    The target gate is specified by the phases :math:`\phi, \theta`.
    Some phases can remain unspecified if they may take on arbitrary values.
    In the figure, we use the notation :math:`\mathrm{C}_1\mathrm{Z}(\alpha) = \mathrm{diag}(1, 1, 1, e^{i\alpha})`,
    and :math:`\mathrm{Z}(\alpha) = \mathrm{diag}(1, e^{i\alpha})`.

    .. image:: ../_static/TwoQubitGate.png

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.TwoQubitGate(
        ...     phi=None,
        ...     theta=np.pi,
        ...     Vnn=float("inf"),
        ...     decay=0.0001,
        ... )

    Args:
        phi: target phase :math:`\phi` of single-qubit gate contribution.
        theta: target phase :math:`\theta` of two-qubit gate contribution.
        Vnn: interaction strength :math:`V_{\mathrm{nn}}/(\hbar\Omega_0)`.
        decay: Rydberg decay strength :math:`\gamma/\Omega_0`.

    Returns:
        Two-qubit gate object.
    """

    def __init__(
        self, phi: float | None, theta: float | None, Vnn: float, decay: float
    ):
        super().__init__(decay)
        self._phi = phi
        self._theta = theta
        self._Vnn = Vnn

    def dim(self) -> int:
        return 4

    def get_gate_angles(self) -> tuple[float | None, float | None]:
        r"""
        Returns:
            Gate phases :math:`\phi, \theta`.
        """
        return self._phi, self._theta

    def get_interactions(self) -> float:
        r"""
        Returns:
            Interaction strength :math:`V_{\mathrm{nn}}/(\hbar\Omega_0)`.
        """
        return self._Vnn

    def subsystem_hamiltonians(self) -> tuple:
        if isinf(float(self._Vnn)):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
            )
        return (
            partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
            partial(H_2_atoms, decay=self._decay, V=self._Vnn),
        )

    def subsystem_rydberg_population_operators(self) -> tuple:
        if isinf(float(self._Vnn)):
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
            )
        return (
            H_k_atoms_perfect_blockade(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1),
            H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
        )

    def initial_states(self) -> tuple:
        if isinf(float(self._Vnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0j]),
        )

    def process_fidelity(self, final_states) -> float:
        # Obtained diagonal gate matrix
        obtained_gate = jnp.array(
            [
                1,
                final_states[0][0],
                final_states[0][0],
                final_states[1][0],
            ]
        )

        # Targeted diagonal gate matrix
        p = jnp.angle(obtained_gate[1]) if self._phi is None else self._phi
        t = jnp.angle(obtained_gate[3]) - 2 * p if self._theta is None else self._theta

        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
            ]
        )

        return (
            jnp.abs(jnp.vdot(targeted_gate, obtained_gate)) ** 2
            / len(targeted_gate) ** 2
        )

    def rydberg_time(self, expectation_values):
        return (1 / 4) * jnp.squeeze(2 * expectation_values[0] + expectation_values[1])
