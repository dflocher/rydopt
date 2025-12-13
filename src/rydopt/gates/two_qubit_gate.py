from __future__ import annotations

from copy import deepcopy
from functools import partial
from math import isinf

import jax.numpy as jnp
from typing_extensions import Self

from rydopt.gates.subsystem_hamiltonians import (
    H_2_atoms,
    H_k_atoms_perfect_blockade,
)
from rydopt.types import HamiltonianFunction


class TwoQubitGate:
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

    """

    def __init__(self, phi: float | None, theta: float | None, Vnn: float, decay: float):
        self._phi = phi
        self._theta = theta
        self._Vnn = Vnn
        self._decay = decay

    def with_decay(self, decay: float) -> Self:
        r"""Creates a copy of the gate with a new decay strength.

        Args:
            decay: New decay strength :math:`\gamma/\Omega_0`.

        Returns:
            A copy of the gate object with the new decay strength.

        """
        new = deepcopy(self)
        new._decay = decay
        return new

    def dim(self) -> int:
        r"""Hilbert space dimension.

        Returns:
            4

        """
        return 4

    def hamiltonian_functions_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        r"""The full gate Hamiltonian can be split into distinct blocks that describe the time evolution
        of basis states. The number of blocks and their dimensionality depends on the interaction strengths.

        Returns:
            Tuple of Hamiltonian functions.

        """
        if isinf(float(self._Vnn)):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
            )
        return (
            partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
            partial(H_2_atoms, decay=self._decay, V=self._Vnn),
        )

    def rydberg_population_operators_for_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""For each basis state, the Rydberg population operators count the number of Rydberg excitations on
        the diagonal.

        Returns:
            Tuple of operators.

        """
        if isinf(float(self._Vnn)):
            return (
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
            )
        return (
            H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
            H_2_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
        )

    def initial_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension are
        provided.

        Returns:
            Tuple of arrays.

        """
        if isinf(float(self._Vnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0j]),
        )

    def process_fidelity(self, final_basis_states) -> jnp.ndarray:
        r"""Given the basis states evolved under the pulse,
        this function calculates the fidelity with respect to the gate's target state, specified by the gate angles
        :math:`\phi, \, \theta, \, \ldots`

        Args:
            final_basis_states: Time-evolved basis states.

        Returns:
            Fidelity with respect to the target state.

        """
        # Obtained diagonal gate matrix
        obtained_gate = jnp.array(
            [
                1,
                final_basis_states[0][0],
                final_basis_states[0][0],
                final_basis_states[1][0],
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

        return jnp.abs(jnp.vdot(targeted_gate, obtained_gate)) ** 2 / len(targeted_gate) ** 2

    def rydberg_time(self, expectation_values_of_basis_states) -> jnp.ndarray:
        r"""Given the expectation values of Rydberg populations for each basis state, integrated over the full
        pulse, this function calculates the average time spent in Rydberg states during the gate.

        Args:
            expectation_values_of_basis_states: Expected Rydberg times for each basis state.

        Returns:
            Averaged Rydberg time :math:`T_R`.

        """
        return (1 / 4) * jnp.squeeze(2 * expectation_values_of_basis_states[0] + expectation_values_of_basis_states[1])
