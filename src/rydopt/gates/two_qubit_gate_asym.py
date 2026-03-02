from __future__ import annotations

import warnings
from copy import deepcopy
from functools import partial
from math import isinf

import jax
import jax.numpy as jnp
from typing_extensions import Self

from rydopt.gates.subsystem_hamiltonians_general import (
    H_1_atom_general,
    H_2_atoms_general,
)
from rydopt.types import HamiltonianFunction


class TwoQubitGateAsym:
    r"""Class that describes a gate on two atoms in an asymmetric setup.

    The physical setting is described by the interaction strength between atoms, :math:`V_{12}`,
    and the decay strength from Rydberg states, :math:`\gamma`. In addition, each atom can optionally
    have a different Rabi frequency scaling factor.

    The target gate is specified by the phases :math:`\phi_1, \phi_2, \theta_{12}`.
    Some phases can remain unspecified if they may take on arbitrary values.

    Args:
        phi1: target phase of the single-qubit gate contribution on atom 1.
        phi2: target phase of the single-qubit gate contribution on atom 2.
        theta12: target phase of the two-qubit gate contribution.
        V12: interaction strength between atoms 1 and 2, :math:`V_{12}/(\hbar\Omega_0)`.
        decay: Rydberg decay strength :math:`\gamma/\Omega_0`, default is 0.
        s1: Rabi frequency scaling factor for atom 1, default is 1.
        s2: Rabi frequency scaling factor for atom 2, default is 1.

    """

    def __init__(
        self,
        phi1: float | None,
        phi2: float | None,
        theta12: float | None,
        V12: float,
        decay: float = 0.0,
        s1: float = 1.0,
        s2: float = 1.0,
    ) -> None:
        if isinf(float(V12)):
            raise ValueError(
                "V12 must be finite. If the setup is symmetric, use `TwoQubitGate` for infinite interaction strengths."
            )

        warnings.warn(
            "This gate implementation does not use any symmetries. If the Rabi frequencies are the "
            "same on both atoms, consider using `TwoQubitGate` for better performance.",
            stacklevel=2,
        )

        self._phi1 = phi1
        self._phi2 = phi2
        self._theta12 = theta12

        self._V12 = V12

        self._decay = decay

        self._s1 = s1
        self._s2 = s2

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
        of basis states.

        Returns:
            Tuple of Hamiltonian functions.

        """
        return (
            # |01>
            partial(H_1_atom_general, decay=self._decay, s1=self._s2),
            # |10>
            partial(H_1_atom_general, decay=self._decay, s1=self._s1),
            # |11>
            partial(
                H_2_atoms_general,
                decay=self._decay,
                V12=self._V12,
                s1=self._s1,
                s2=self._s2,
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
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
        )

    def initial_basis_states(self) -> tuple[jax.Array, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension are
        provided.

        Returns:
            Tuple of arrays.

        """
        z2 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
        z4 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
        return (z2, z2, z4)

    def process_fidelity(self, final_basis_states: tuple[jax.Array, ...]) -> jax.Array:
        r"""Given the basis states evolved under the pulse,
        this function calculates the fidelity with respect to the gate's target state.

        Args:
            final_basis_states: Time-evolved basis states.

        Returns:
            Fidelity with respect to the target state.

        """
        # Obtained diagonal gate matrix
        obtained_gate = jnp.array(
            [
                1,  # 0: |00>
                final_basis_states[0][0],  # 1: |01>
                final_basis_states[1][0],  # 2: |10>
                final_basis_states[2][0],  # 3: |11>
            ]
        )

        # Single-qubit phases
        p1 = jnp.angle(obtained_gate[2]) if self._phi1 is None else self._phi1
        p2 = jnp.angle(obtained_gate[1]) if self._phi2 is None else self._phi2

        # Two-qubit phase
        t12 = jnp.angle(obtained_gate[3]) - p1 - p2 if self._theta12 is None else self._theta12

        # Targeted diagonal gate matrix
        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p2),
                jnp.exp(1j * p1),
                jnp.exp(1j * (p1 + p2 + t12)),
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
        return (1 / 4) * jnp.squeeze(
            expectation_values_of_basis_states[0]
            + expectation_values_of_basis_states[1]
            + expectation_values_of_basis_states[2]
        )
