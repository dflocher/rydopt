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
    H_3_atoms_general,
    H_4_atoms_general,  # must exist in your package, analogous to H_3_atoms_general
)
from rydopt.types import HamiltonianFunction


class FourQubitGateAsym:
    r"""Class that describes a gate on four atoms in an asymmetric setup.

    The physical setting is described by the interaction strengths between atoms,
    :math:`V_{12}, V_{13}, V_{14}, V_{23}, V_{24}, V_{34}`, and the decay strength from
    Rydberg states, :math:`\gamma`. In addition, each atom can optionally have a different
    Rabi frequency scaling factor.
    The target gate is specified by phases
    :math:`\phi_1, \phi_2, \phi_3, \phi_4`,
    :math:`\theta_{12}, \theta_{13}, \theta_{14}, \theta_{23}, \theta_{24}, \theta_{34}`,
    :math:`\lambda_{123}, \lambda_{124}, \lambda_{134}, \lambda_{234}`,
    and :math:`\mu`.
    Some phases can remain unspecified if they may take on arbitrary values.

    Args:
        phi1: target phase of the single-qubit gate contribution on atom 1.
        phi2: target phase of the single-qubit gate contribution on atom 2.
        phi3: target phase of the single-qubit gate contribution on atom 3.
        phi4: target phase of the single-qubit gate contribution on atom 4.
        theta12: target phase of the two-qubit gate contribution on atoms 1, 2.
        theta13: target phase of the two-qubit gate contribution on atoms 1, 3.
        theta14: target phase of the two-qubit gate contribution on atoms 1, 4.
        theta23: target phase of the two-qubit gate contribution on atoms 2, 3.
        theta24: target phase of the two-qubit gate contribution on atoms 2, 4.
        theta34: target phase of the two-qubit gate contribution on atoms 3, 4.
        lamb123: target phase of the three-qubit gate contribution on atoms 1, 2, 3.
        lamb124: target phase of the three-qubit gate contribution on atoms 1, 2, 4.
        lamb134: target phase of the three-qubit gate contribution on atoms 1, 3, 4.
        lamb234: target phase of the three-qubit gate contribution on atoms 2, 3, 4.
        mu: target phase of the four-qubit gate contribution.
        V12: interaction strength between atoms 1 and 2, :math:`V_{12}/(\hbar\Omega_0)`.
        V13: interaction strength between atoms 1 and 3, :math:`V_{13}/(\hbar\Omega_0)`.
        V14: interaction strength between atoms 1 and 4, :math:`V_{14}/(\hbar\Omega_0)`.
        V23: interaction strength between atoms 2 and 3, :math:`V_{23}/(\hbar\Omega_0)`.
        V24: interaction strength between atoms 2 and 4, :math:`V_{24}/(\hbar\Omega_0)`.
        V34: interaction strength between atoms 3 and 4, :math:`V_{34}/(\hbar\Omega_0)`.
        decay: Rydberg decay strength :math:`\gamma/\Omega_0`, default is 0.
        s1: Rabi frequency scaling factor for atom 1, default is 1.
        s2: Rabi frequency scaling factor for atom 2, default is 1.
        s3: Rabi frequency scaling factor for atom 3, default is 1.
        s4: Rabi frequency scaling factor for atom 4, default is 1.

    """

    def __init__(
        self,
        phi1: float | None,
        phi2: float | None,
        phi3: float | None,
        phi4: float | None,
        theta12: float | None,
        theta13: float | None,
        theta14: float | None,
        theta23: float | None,
        theta24: float | None,
        theta34: float | None,
        lamb123: float | None,
        lamb124: float | None,
        lamb134: float | None,
        lamb234: float | None,
        mu: float | None,
        V12: float,
        V13: float,
        V14: float,
        V23: float,
        V24: float,
        V34: float,
        decay: float = 0.0,
        s1: float = 1.0,
        s2: float = 1.0,
        s3: float = 1.0,
        s4: float = 1.0,
    ) -> None:
        for name, val in [("V12", V12), ("V13", V13), ("V14", V14), ("V23", V23), ("V24", V24), ("V34", V34)]:
            if isinf(float(val)):
                raise ValueError(
                    f"{name} must be finite. If the setup is symmetric, use `FourQubitGatePyramidal` "
                    "for infinite interaction strengths."
                )

        warnings.warn(
            "This gate implementation does not use any symmetries. If your setup is a pyramidal arrangement of atoms, "
            "consider using `FourQubitGatePyramidal` for better performance.",
            stacklevel=2,
        )

        self._phi1 = phi1
        self._phi2 = phi2
        self._phi3 = phi3
        self._phi4 = phi4
        self._theta12 = theta12
        self._theta13 = theta13
        self._theta14 = theta14
        self._theta23 = theta23
        self._theta24 = theta24
        self._theta34 = theta34
        self._lamb123 = lamb123
        self._lamb124 = lamb124
        self._lamb134 = lamb134
        self._lamb234 = lamb234
        self._mu = mu

        self._V12 = V12
        self._V13 = V13
        self._V14 = V14
        self._V23 = V23
        self._V24 = V24
        self._V34 = V34

        self._decay = decay

        self._s1 = s1
        self._s2 = s2
        self._s3 = s3
        self._s4 = s4

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
            16

        """
        return 16

    def hamiltonian_functions_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        r"""The full gate Hamiltonian can be split into distinct blocks that describe the time evolution
        of basis states.

        Returns:
            Tuple of Hamiltonian functions.

        """
        return (
            # |0001>
            partial(H_1_atom_general, decay=self._decay, s1=self._s4),
            # |0010>
            partial(H_1_atom_general, decay=self._decay, s1=self._s3),
            # |0011>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V34, s1=self._s3, s2=self._s4),
            # |0100>
            partial(H_1_atom_general, decay=self._decay, s1=self._s2),
            # |0101>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V24, s1=self._s2, s2=self._s4),
            # |0110>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V23, s1=self._s2, s2=self._s3),
            # |0111>
            partial(
                H_3_atoms_general,
                decay=self._decay,
                V12=self._V23,
                V13=self._V24,
                V23=self._V34,
                s1=self._s2,
                s2=self._s3,
                s3=self._s4,
            ),
            # |1000>
            partial(H_1_atom_general, decay=self._decay, s1=self._s1),
            # |1001>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V14, s1=self._s1, s2=self._s4),
            # |1010>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V13, s1=self._s1, s2=self._s3),
            # |1011>
            partial(
                H_3_atoms_general,
                decay=self._decay,
                V12=self._V13,
                V13=self._V14,
                V23=self._V34,
                s1=self._s1,
                s2=self._s3,
                s3=self._s4,
            ),
            # |1100>
            partial(H_2_atoms_general, decay=self._decay, V12=self._V12, s1=self._s1, s2=self._s2),
            # |1101>
            partial(
                H_3_atoms_general,
                decay=self._decay,
                V12=self._V12,
                V13=self._V14,
                V23=self._V24,
                s1=self._s1,
                s2=self._s2,
                s3=self._s4,
            ),
            # |1110>
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
            # |1111>
            partial(
                H_4_atoms_general,
                decay=self._decay,
                V12=self._V12,
                V13=self._V13,
                V14=self._V14,
                V23=self._V23,
                V24=self._V24,
                V34=self._V34,
                s1=self._s1,
                s2=self._s2,
                s3=self._s3,
                s4=self._s4,
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
            H_1_atom_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0),
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
            H_3_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0, V13=0.0, V23=0.0),
            H_1_atom_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0),
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
            H_3_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0, V13=0.0, V23=0.0),
            H_2_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0),
            H_3_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0, V13=0.0, V23=0.0),
            H_3_atoms_general(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0, V13=0.0, V23=0.0),
            H_4_atoms_general(
                Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V12=0.0, V13=0.0, V14=0.0, V23=0.0, V24=0.0, V34=0.0
            ),
        )

    def initial_basis_states(self) -> tuple[jax.Array, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension are
        provided.

        Returns:
            Tuple of arrays.

        """
        z2 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
        z4 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
        z8 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
        z16 = jnp.array(
            [
                1.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        )

        return (z2, z2, z4, z2, z4, z4, z8, z2, z4, z4, z8, z4, z8, z8, z16)

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
                1,  # 0: |0000>
                final_basis_states[0][0],  # 1: |0001>
                final_basis_states[1][0],  # 2: |0010>
                final_basis_states[2][0],  # 3: |0011>
                final_basis_states[3][0],  # 4: |0100>
                final_basis_states[4][0],  # 5: |0101>
                final_basis_states[5][0],  # 6: |0110>
                final_basis_states[6][0],  # 7: |0111>
                final_basis_states[7][0],  # 8: |1000>
                final_basis_states[8][0],  # 9: |1001>
                final_basis_states[9][0],  # 10: |1010>
                final_basis_states[10][0],  # 11: |1011>
                final_basis_states[11][0],  # 12: |1100>
                final_basis_states[12][0],  # 13: |1101>
                final_basis_states[13][0],  # 14: |1110>
                final_basis_states[14][0],  # 15: |1111>
            ]
        )

        # Single-qubit phase (averaged)
        p1 = jnp.angle(obtained_gate[8]) if self._phi1 is None else self._phi1
        p2 = jnp.angle(obtained_gate[4]) if self._phi2 is None else self._phi2
        p3 = jnp.angle(obtained_gate[2]) if self._phi3 is None else self._phi3
        p4 = jnp.angle(obtained_gate[1]) if self._phi4 is None else self._phi4

        # Two-qubit phases
        t12 = jnp.angle(obtained_gate[12]) - p1 - p2 if self._theta12 is None else self._theta12
        t13 = jnp.angle(obtained_gate[10]) - p1 - p3 if self._theta13 is None else self._theta13
        t14 = jnp.angle(obtained_gate[9]) - p1 - p4 if self._theta14 is None else self._theta14
        t23 = jnp.angle(obtained_gate[6]) - p2 - p3 if self._theta23 is None else self._theta23
        t24 = jnp.angle(obtained_gate[5]) - p2 - p4 if self._theta24 is None else self._theta24
        t34 = jnp.angle(obtained_gate[3]) - p3 - p4 if self._theta34 is None else self._theta34

        # Three-qubit phases
        l234 = jnp.angle(obtained_gate[7]) - p2 - p3 - p4 - t23 - t24 - t34 if self._lamb234 is None else self._lamb234
        l134 = jnp.angle(obtained_gate[11]) - p1 - p3 - p4 - t13 - t14 - t34 if self._lamb134 is None else self._lamb134
        l124 = jnp.angle(obtained_gate[13]) - p1 - p2 - p4 - t12 - t14 - t24 if self._lamb124 is None else self._lamb124
        l123 = jnp.angle(obtained_gate[14]) - p1 - p2 - p3 - t12 - t13 - t23 if self._lamb123 is None else self._lamb123

        # Four-qubit phase
        mu = (
            jnp.angle(obtained_gate[15])
            - p1
            - p2
            - p3
            - p4
            - t12
            - t13
            - t14
            - t23
            - t24
            - t34
            - l123
            - l124
            - l134
            - l234
            if self._mu is None
            else self._mu
        )

        # Targeted diagonal gate matrix
        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p4),
                jnp.exp(1j * p3),
                jnp.exp(1j * (p3 + p4 + t34)),
                jnp.exp(1j * p2),
                jnp.exp(1j * (p2 + p4 + t24)),
                jnp.exp(1j * (p2 + p3 + t23)),
                jnp.exp(1j * (p2 + p3 + p4 + t23 + t24 + t34 + l234)),
                jnp.exp(1j * p1),
                jnp.exp(1j * (p1 + p4 + t14)),
                jnp.exp(1j * (p1 + p3 + t13)),
                jnp.exp(1j * (p1 + p3 + p4 + t13 + t14 + t34 + l134)),
                jnp.exp(1j * (p1 + p2 + t12)),
                jnp.exp(1j * (p1 + p2 + p4 + t12 + t14 + t24 + l124)),
                jnp.exp(1j * (p1 + p2 + p3 + t12 + t13 + t23 + l123)),
                jnp.exp(1j * (p1 + p2 + p3 + p4 + t12 + t13 + t14 + t23 + t24 + t34 + l123 + l124 + l134 + l234 + mu)),
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
        return (1 / 16) * jnp.squeeze(
            expectation_values_of_basis_states[0]
            + expectation_values_of_basis_states[1]
            + expectation_values_of_basis_states[2]
            + expectation_values_of_basis_states[3]
            + expectation_values_of_basis_states[4]
            + expectation_values_of_basis_states[5]
            + expectation_values_of_basis_states[6]
            + expectation_values_of_basis_states[7]
            + expectation_values_of_basis_states[8]
            + expectation_values_of_basis_states[9]
            + expectation_values_of_basis_states[10]
            + expectation_values_of_basis_states[11]
            + expectation_values_of_basis_states[12]
            + expectation_values_of_basis_states[13]
            + expectation_values_of_basis_states[14]
        )
