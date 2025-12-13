from __future__ import annotations

from copy import deepcopy
from functools import partial
from math import isinf

import jax.numpy as jnp
from typing_extensions import Self

from rydopt.gates.subsystem_hamiltonians import (
    H_2_atoms,
    H_3_atoms,
    H_3_atoms_inf_V,
    H_3_atoms_symmetric,
    H_4_atoms,
    H_4_atoms_inf_V,
    H_4_atoms_symmetric,
    H_k_atoms_perfect_blockade,
)
from rydopt.types import HamiltonianFunction


class FourQubitGatePyramidal:
    r"""Class that describes a gate on four atoms arranged in a pyramid.
    The physical setting is described by the interaction strengths between atoms, :math:`V_{\mathrm{nn}}`
    and :math:`V_{\mathrm{nnn}}`, and the decay strength from Rydberg states, :math:`\gamma`.
    The target gate is specified by the phases :math:`\phi, \theta, \theta', \lambda, \lambda', \kappa`.
    Some phases can remain unspecified if they may take on arbitrary values.
    In the figure, we use the notation :math:`\mathrm{C}_n\mathrm{Z}(\alpha) = \mathrm{diag}(1, ..., 1, e^{i\alpha})`
    on :math:`n+1` qubits, and
    :math:`\mathrm{Z}(\alpha) = \mathrm{C}_0\mathrm{Z}(\alpha) = \mathrm{diag}(1, e^{i\alpha})`.

    .. image:: ../_static/FourQubitGatePyramidal.png

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.FourQubitGatePyramidal(
        ...     phi=None,
        ...     theta=np.pi,
        ...     theta_prime=None,
        ...     lamb=0.0,
        ...     lamb_prime=None,
        ...     kappa=np.pi,
        ...     Vnn=float("inf"),
        ...     Vnnn=1.0,
        ...     decay=0.0001,
        ... )

    Args:
        phi: target phase :math:`\phi` of single-qubit gate contribution.
        theta: target phase :math:`\theta` of nearest-neighbour two-qubit gate contribution.
        theta_prime: target phase :math:`\theta'` of next-nearest-neighbour two-qubit gate contribution.
        lamb: target phase :math:`\lambda` of asymmetric three-qubit gate contribution.
        lamb_prime: target phase :math:`\lambda'` of symmetric three-qubit gate contribution.
        kappa: target phase :math:`\kappa` of four-qubit gate contribution.
        Vnn: nearest-neighbour interaction strength :math:`V_{\mathrm{nn}}/(\hbar\Omega_0)`.
        Vnnn: next-nearest-neighbour interaction strength :math:`V_{\mathrm{nnn}}/(\hbar\Omega_0)`.
        decay: Rydberg decay strength :math:`\gamma/\Omega_0`.

    """

    def __init__(
        self,
        phi: float | None,
        theta: float | None,
        theta_prime: float | None,
        lamb: float | None,
        lamb_prime: float | None,
        kappa: float | None,
        Vnn: float,
        Vnnn: float,
        decay: float,
    ):
        if (Vnn == Vnnn) and ((theta != theta_prime) or (lamb != lamb_prime)):
            raise ValueError("For Vnn=Vnnn, theta=theta_prime and lambda=lamb_prime is required")
        if (Vnnn == 0) and ((theta_prime != 0.0) or (lamb_prime != 0.0)):
            raise ValueError("For Vnnn=0, theta_prime=0 and lamb_prime=0 is required")
        self._phi = phi
        self._theta = theta
        self._theta_prime = theta_prime
        self._lamb = lamb
        self._lamb_prime = lamb_prime
        self._kappa = kappa
        self._Vnn = Vnn
        self._Vnnn = Vnnn
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
            16

        """
        return 16

    def hamiltonian_functions_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        r"""The full gate Hamiltonian can be split into distinct blocks that describe the time evolution
        of basis states. The number of blocks and their dimensionality depends on the interaction strengths.

        Returns:
            Tuple of Hamiltonian functions.

        """
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=3),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=4),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_2_atoms, decay=self._decay, V=self._Vnn),
                partial(H_3_atoms_symmetric, decay=self._decay, V=self._Vnn),
                partial(H_4_atoms_symmetric, decay=self._decay, V=self._Vnn),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_3_atoms_inf_V, decay=self._decay, V=self._Vnnn),
                partial(H_4_atoms_inf_V, decay=self._decay, V=self._Vnnn),
            )
        if isinf(float(self._Vnn)):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_2_atoms, decay=self._decay, V=self._Vnnn),
                partial(H_3_atoms_inf_V, decay=self._decay, V=self._Vnnn),
                partial(H_3_atoms_symmetric, decay=self._decay, V=self._Vnnn),
                partial(H_4_atoms_inf_V, decay=self._decay, V=self._Vnnn),
            )
        return (
            partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
            partial(H_2_atoms, decay=self._decay, V=self._Vnn),
            partial(H_2_atoms, decay=self._decay, V=self._Vnnn),
            partial(H_3_atoms, decay=self._decay, Vnn=self._Vnn, Vnnn=self._Vnnn),
            partial(H_3_atoms_symmetric, decay=self._decay, V=self._Vnnn),
            partial(H_4_atoms, decay=self._decay, Vnn=self._Vnn, Vnnn=self._Vnnn),
        )

    def rydberg_population_operators_for_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""For each basis state, the Rydberg population operators count the number of Rydberg excitations on
        the diagonal.

        Returns:
            Tuple of operators.

        """
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_2_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_3_atoms_symmetric(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_4_atoms_symmetric(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_3_atoms_inf_V(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_4_atoms_inf_V(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        if isinf(float(self._Vnn)):
            return (
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
                H_2_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_3_atoms_inf_V(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_3_atoms_symmetric(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_4_atoms_inf_V(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        return (
            H_k_atoms_perfect_blockade(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, k=1),
            H_2_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_2_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_3_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, Vnn=0.0, Vnnn=0.0),
            H_3_atoms_symmetric(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_4_atoms(Delta=1.0, Xi=0.0, Omega=0.0, decay=0.0, Vnn=0.0, Vnnn=0.0),
        )

    def initial_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension are
        provided.

        Returns:
            Tuple of arrays.

        """
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [
                    1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ]
            ),
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
        if float(self._Vnn) == float(self._Vnnn):
            obtained_gate = jnp.array(
                [
                    1,
                    final_basis_states[0][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[1][0],
                    final_basis_states[2][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[1][0],
                    final_basis_states[2][0],
                    final_basis_states[1][0],
                    final_basis_states[2][0],
                    final_basis_states[2][0],
                    final_basis_states[3][0],
                ]
            )
        elif isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            obtained_gate = jnp.array(
                [
                    1,
                    final_basis_states[0][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[0][0] ** 2,
                    final_basis_states[2][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[0][0] ** 2,
                    final_basis_states[2][0],
                    final_basis_states[0][0] ** 2,
                    final_basis_states[2][0],
                    final_basis_states[0][0] ** 3,
                    final_basis_states[3][0],
                ]
            )
        else:
            obtained_gate = jnp.array(
                [
                    1,
                    final_basis_states[0][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[2][0],
                    final_basis_states[3][0],
                    final_basis_states[0][0],
                    final_basis_states[1][0],
                    final_basis_states[2][0],
                    final_basis_states[3][0],
                    final_basis_states[2][0],
                    final_basis_states[3][0],
                    final_basis_states[4][0],
                    final_basis_states[5][0],
                ]
            )

        # Targeted diagonal gate matrix
        p = jnp.angle(obtained_gate[1]) if self._phi is None else self._phi
        t = jnp.angle(obtained_gate[3]) - 2 * p if self._theta is None else self._theta
        e = jnp.angle(obtained_gate[6]) - 2 * p if self._theta_prime is None else self._theta_prime
        l = jnp.angle(obtained_gate[7]) - 3 * p - 2 * t - e if self._lamb is None else self._lamb
        d = jnp.angle(obtained_gate[14]) - 3 * p - 3 * e if self._lamb_prime is None else self._lamb_prime
        k = jnp.angle(obtained_gate[15]) - 4 * p - 3 * t - 3 * e - 3 * l - d if self._kappa is None else self._kappa

        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * (3 * p + 2 * t + e + l)),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * (3 * p + 2 * t + e + l)),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * (3 * p + 2 * t + e + l)),
                jnp.exp(1j * (3 * p + 3 * e + d)),
                jnp.exp(1j * (4 * p + 3 * t + 3 * e + 3 * l + d + k)),
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
        if float(self._Vnn) == float(self._Vnnn):
            return (1 / 16) * jnp.squeeze(
                4 * expectation_values_of_basis_states[0]
                + 6 * expectation_values_of_basis_states[1]
                + 4 * expectation_values_of_basis_states[2]
                + expectation_values_of_basis_states[3]
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (1 / 16) * jnp.squeeze(
                13 * expectation_values_of_basis_states[0]
                + 3 * expectation_values_of_basis_states[1]
                + 3 * expectation_values_of_basis_states[2]
                + expectation_values_of_basis_states[3]
            )
        return (1 / 16) * jnp.squeeze(
            4 * expectation_values_of_basis_states[0]
            + 3 * expectation_values_of_basis_states[1]
            + 3 * expectation_values_of_basis_states[2]
            + 3 * expectation_values_of_basis_states[3]
            + expectation_values_of_basis_states[4]
            + expectation_values_of_basis_states[5]
        )
