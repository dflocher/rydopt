from __future__ import annotations

from copy import deepcopy
from functools import partial

import jax.numpy as jnp
from typing_extensions import Self

from rydopt.gates.subsystem_hamiltonians import (
    H_2_atoms,
    H_3_atoms_asym,
    H_k_atoms_perfect_blockade,
)
from rydopt.types import HamiltonianFunction


class ThreeQubitGate:  # ToDo: merge ThreeQubitGate and ThreeQubitGateIsosceles?
    r"""Class that describes a gate on three atoms.
    The physical setting is described by the interaction strengths between atoms, :math:`V_{1}`,
    :math:`V_{2}`, and :math:`V_{3}`, and the decay strength from Rydberg states, :math:`\gamma`.
    The target gate is specified by the phases :math:`\phi, \theta, \theta', \theta'', \lambda`.
    Some phases can remain unspecified if they may take on arbitrary values.

    Args:
        phi: target phase :math:`\phi` of single-qubit gate contribution.
        theta: target phase :math:`\theta` of 1-2 two-qubit gate contribution.
        theta_prime: target phase :math:`\theta'` of 2-3 two-qubit gate contribution.
        theta_pprime: target phase :math:`\theta''` of 3-1 two-qubit gate contribution.
        lamb: target phase :math:`\lambda` of three-qubit gate contribution.
        V1: interaction strength between atoms 1 and 2, :math:`V_{1}/(\hbar\Omega_0)`.
        V2: interaction strength between atoms 2 and 3, :math:`V_{2}/(\hbar\Omega_0)`.
        V3: interaction strength between atoms 3 and 1, :math:`V_{3}/(\hbar\Omega_0)`.
        decay: Rydberg decay strength :math:`\gamma/\Omega_0`.

    """

    def __init__(
        self,
        phi: float | None,
        theta: float | None,
        theta_prime: float | None,
        theta_pprime: float | None,
        lamb: float | None,
        V1: float,
        V2: float,
        V3: float,
        decay: float,
    ) -> None:
        self._phi = phi
        self._theta = theta
        self._theta_prime = theta_prime
        self._theta_pprime = theta_pprime
        self._lamb = lamb
        self._V1 = V1
        self._V2 = V2
        self._V3 = V3
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
            8

        """
        return 8

    def hamiltonian_functions_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        r"""The full gate Hamiltonian can be split into distinct blocks that describe the time evolution
        of basis states. The number of blocks and their dimensionality depends on the interaction strengths.

        Returns:
            Tuple of Hamiltonian functions.

        """
        return (
            partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
            partial(H_2_atoms, decay=self._decay, V=self._V2),
            partial(H_2_atoms, decay=self._decay, V=self._V3),
            partial(H_2_atoms, decay=self._decay, V=self._V1),
            partial(H_3_atoms_asym, decay=self._decay, V1=self._V1, V2=self._V2, V3=self._V3),
        )

    def rydberg_population_operators_for_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""For each basis state, the Rydberg population operators count the number of Rydberg excitations on
        the diagonal.

        Returns:
            Tuple of operators.

        """
        return (
            H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
            H_2_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_2_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_2_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_3_atoms_asym(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V1=0.0, V2=0.0, V3=0.0),
        )

    def initial_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension are
        provided.

        Returns:
            Tuple of arrays.

        """
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        )

    def process_fidelity(self, final_basis_states: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
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
                final_basis_states[0][0],
                final_basis_states[2][0],
                final_basis_states[3][0],
                final_basis_states[4][0],
            ]
        )

        # Targeted diagonal gate matrix
        p = jnp.angle(obtained_gate[1]) if self._phi is None else self._phi
        t = jnp.angle(obtained_gate[6]) - 2 * p if self._theta is None else self._theta
        e = jnp.angle(obtained_gate[3]) - 2 * p if self._theta_prime is None else self._theta_prime
        f = jnp.angle(obtained_gate[5]) - 2 * p if self._theta_pprime is None else self._theta_pprime
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

    def rydberg_time(self, expectation_values_of_basis_states: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        r"""Given the expectation values of Rydberg populations for each basis state, integrated over the full
        pulse, this function calculates the average time spent in Rydberg states during the gate.

        Args:
            expectation_values_of_basis_states: Expected Rydberg times for each basis state.

        Returns:
            Averaged Rydberg time :math:`T_R`.

        """
        return (1 / 8) * jnp.squeeze(
            3 * expectation_values_of_basis_states[0]
            + expectation_values_of_basis_states[1]
            + expectation_values_of_basis_states[2]
            + expectation_values_of_basis_states[3]
            + expectation_values_of_basis_states[4]
        )
