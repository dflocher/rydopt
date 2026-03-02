from __future__ import annotations

import warnings
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
from typing_extensions import Self

from rydopt.gates.subsystem_hamiltonians_general import (
    H_1_atom_general,
    H_2_atoms_general,
    H_3_atoms_general,
)
from rydopt.types import HamiltonianFunction


class ThreeQubitGateAsym:
    r"""Class that describes a gate on three atoms in an asymmetric setup.

    The physical setting is described by the interaction strengths between atoms, :math:`V_{12}`,
    :math:`V_{13}`, and :math:`V_{23}`, and the decay strength from Rydberg states, :math:`\gamma`.
    In addition, each atom can optionally have a different Rabi frequency scaling factor.
    The target gate is specified by the phases :math:`\phi, \theta_{12}, \theta_{13}, \theta_{23}, \lambda`.
    Some phases can remain unspecified if they may take on arbitrary values.

    Args:
        phi: target phase :math:`\phi` of single-qubit gate contribution.
        theta12: target phase :math:`\theta_{12}` of the 1-2 two-qubit gate contribution.
        theta13: target phase :math:`\theta_{13}` of the 1-3 two-qubit gate contribution.
        theta23: target phase :math:`\theta_{23}` of the 2-3 two-qubit gate contribution.
        lamb: target phase :math:`\lambda` of three-qubit gate contribution.
        V12: interaction strength between atoms 1 and 2, :math:`V_{12}/(\hbar\Omega_0)`.
        V13: interaction strength between atoms 1 and 3, :math:`V_{13}/(\hbar\Omega_0)`.
        V23: interaction strength between atoms 2 and 3, :math:`V_{23}/(\hbar\Omega_0)`.
        decay: Rydberg decay strength :math:`\gamma/\Omega_0`, default is 0.
        s1: Rabi frequency scaling factor for atom 1, default is 1.
        s2: Rabi frequency scaling factor for atom 2, default is 1.
        s3: Rabi frequency scaling factor for atom 3, default is 1.

    """

    def __init__(
        self,
        phi: float | None,
        theta12: float | None,
        theta13: float | None,
        theta23: float | None,
        lamb: float | None,
        V12: float,
        V13: float,
        V23: float,
        decay: float = 0.0,
        s1: float = 1.0,
        s2: float = 1.0,
        s3: float = 1.0,
    ) -> None:
        warnings.warn(
            "This gate implementation does not use any symmetries. If your setup is an isosceles triangle, "
            "consider using `ThreeQubitGateIsosceles` for better performance.",
            stacklevel=2,
        )

        self._phi = phi
        self._theta12 = theta12
        self._theta13 = theta13
        self._theta23 = theta23
        self._lamb = lamb
        self._V12 = V12
        self._V13 = V13
        self._V23 = V23
        self._decay = decay
        self._s1 = s1
        self._s2 = s2
        self._s3 = s3

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
            8

        """
        return 8

    def hamiltonian_functions_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        r"""The full gate Hamiltonian can be split into distinct blocks that describe the time evolution
        of basis states.

        Returns:
            Tuple of Hamiltonian functions.

        """
        return (
            # |100>
            partial(H_1_atom_general, decay=self._decay, s1=self._s1),
            # |010>
            partial(H_1_atom_general, decay=self._decay, s1=self._s2),
            # |001>
            partial(H_1_atom_general, decay=self._decay, s1=self._s3),
            # |110>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V23, s1=self._s2, s2=self._s3),
            # |101>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V13, s1=self._s1, s2=self._s3),
            # |011>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V12, s1=self._s1, s2=self._s2),
            # |111>
            partial(
                H_3_atoms_general,
                decay=self._decay,
                V12=self._V12,
                V13=self._V13,
                V23=self._V23,
                s1=self._s1,
                s2=self._s2,
                s3=self._s3,
            ),
        )

    def rydberg_population_operators_for_basis_states(self) -> tuple[jax.Array, ...]:
        r"""For each basis state, the Rydberg population operators count the number of Rydberg excitations on
        the diagonal.

        Returns:
            Tuple of operators.

        """
        return (
            H_1_atom_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0),
            H_1_atom_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0),
            H_1_atom_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0),
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
            H_3_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0, V23=0.0, V13=0.0),
        )

    def initial_basis_states(self) -> tuple[jax.Array, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension are
        provided.

        Returns:
            Tuple of arrays.

        """
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        )

    def process_fidelity(self, final_basis_states: tuple[jax.Array, ...]) -> jax.Array:
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
                final_basis_states[1][0],
                final_basis_states[3][0],
                final_basis_states[2][0],
                final_basis_states[4][0],
                final_basis_states[5][0],
                final_basis_states[6][0],
            ]
        )

        # Targeted diagonal gate matrix
        p = jnp.angle(obtained_gate[1]) if self._phi is None else self._phi
        t = jnp.angle(obtained_gate[6]) - 2 * p if self._theta12 is None else self._theta12
        e = jnp.angle(obtained_gate[3]) - 2 * p if self._theta23 is None else self._theta23
        f = jnp.angle(obtained_gate[5]) - 2 * p if self._theta13 is None else self._theta13
        l = jnp.angle(obtained_gate[7]) - 3 * p - 2 * t - e if self._lamb is None else self._lamb

        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + f)),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * (3 * p + t + e + f + l)),
            ]
        )

        return jnp.abs(jnp.vdot(targeted_gate, obtained_gate)) ** 2 / len(targeted_gate) ** 2

    def rydberg_time(self, expectation_values_of_basis_states: tuple[jax.Array, ...]) -> jax.Array:
        r"""Given the expectation values of Rydberg populations for each basis state, integrated over the full
        pulse, this function calculates the average time spent in Rydberg states during the gate.

        Args:
            expectation_values_of_basis_states: Expected Rydberg times for each basis state.

        Returns:
            Averaged Rydberg time :math:`T_R`.

        """
        return (1 / 8) * jnp.squeeze(
            expectation_values_of_basis_states[0]
            + expectation_values_of_basis_states[1]
            + expectation_values_of_basis_states[2]
            + expectation_values_of_basis_states[3]
            + expectation_values_of_basis_states[4]
            + expectation_values_of_basis_states[5]
            + expectation_values_of_basis_states[6]
        )
