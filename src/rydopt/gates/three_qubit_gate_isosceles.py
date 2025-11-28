from __future__ import annotations

from functools import partial
import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_k_atoms_perfect_blockade,
    H_2_atoms,
    H_3_atoms_inf_V,
    H_3_atoms_symmetric,
    H_3_atoms,
)
from math import isinf
from rydopt.types import HamiltonianFunction


class ThreeQubitGateIsosceles(Gate):
    r"""Class that describes a gate on three atoms arranged in an isosceles triangle.
    The physical setting is described by the interaction strengths between atoms, :math:`V_{\mathrm{nn}}` and :math:`V_{\mathrm{nnn}}`,
    and the decay strength from Rydberg states, :math:`\gamma`.
    The target gate is specified by the phases :math:`\phi, \theta, \theta', \lambda`.
    Some phases can remain unspecified if they may take on arbitrary values.
    In the figure, we use the notation :math:`\mathrm{C}_n\mathrm{Z}(\alpha) = \mathrm{diag}(1, ..., 1, e^{i\alpha})` on :math:`n+1` qubits,
    and :math:`\mathrm{Z}(\alpha) = \mathrm{C}_0\mathrm{Z}(\alpha) = \mathrm{diag}(1, e^{i\alpha})`.

    .. image:: ../_static/ThreeQubitGateIsosceles.png

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.ThreeQubitGateIsosceles(
        ...     phi=None,
        ...     theta=np.pi,
        ...     theta_prime=None,
        ...     lamb=0.0,
        ...     Vnn=float("inf"),
        ...     Vnnn=1.0,
        ...     decay=0.0001,
        ... )

    Args:
        phi: target phase :math:`\phi` of single-qubit gate contribution.
        theta: target phase :math:`\theta` of nearest-neighbour two-qubit gate contribution.
        theta_prime: target phase :math:`\theta'` of next-nearest-neighbour two-qubit gate contribution.
        lamb: target phase :math:`\lambda` of three-qubit gate contribution.
        Vnn: nearest-neighbour interaction strength :math:`V_{\mathrm{nn}}/(\hbar\Omega_0)`.
        Vnnn: next-nearest-neighbour interaction strength :math:`V_{\mathrm{nnn}}/(\hbar\Omega_0)`.
        decay: Rydberg decay strength :math:`\gamma/\Omega_0`.

    Returns:
        Three-qubit gate object.
    """

    def __init__(
        self,
        phi: float | None,
        theta: float | None,
        theta_prime: float | None,
        lamb: float | None,
        Vnn: float,
        Vnnn: float,
        decay: float,
    ):
        super().__init__(decay)
        if (Vnn == Vnnn) and (theta != theta_prime):
            raise ValueError("For Vnn=Vnnn, theta=theta_prime is required")
        if (Vnnn == 0) and (theta_prime != 0.0):
            raise ValueError("For Vnnn=0, theta_prime=0 is required")
        self._phi = phi
        self._theta = theta
        self._theta_prime = theta_prime
        self._lamb = lamb
        self._Vnn = Vnn
        self._Vnnn = Vnnn

    def dim(self) -> int:
        r"""
        Returns:
            8
        """
        return 8

    def get_gate_angles(
        self,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        r"""
        Returns:
            Gate phases :math:`\phi, \theta, \theta', \lambda`.
        """
        return self._phi, self._theta, self._theta_prime, self._lamb

    def get_interactions(self) -> tuple[float, float]:
        r"""
        Returns:
            Interaction strengths :math:`V_{\mathrm{nn}}/(\hbar\Omega_0), V_{\mathrm{nnn}}/(\hbar\Omega_0)`.
        """
        return self._Vnn, self._Vnnn

    def subsystem_hamiltonians(self) -> tuple[HamiltonianFunction, ...]:
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=3),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_2_atoms, decay=self._decay, V=self._Vnn),
                partial(H_3_atoms_symmetric, decay=self._decay, V=self._Vnn),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_3_atoms_inf_V, decay=self._decay, V=self._Vnnn),
            )
        if isinf(float(self._Vnn)):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_2_atoms, decay=self._decay, V=self._Vnnn),
                partial(H_3_atoms_inf_V, decay=self._decay, V=self._Vnnn),
            )
        return (
            partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
            partial(H_2_atoms, decay=self._decay, V=self._Vnn),
            partial(H_2_atoms, decay=self._decay, V=self._Vnnn),
            partial(H_3_atoms, decay=self._decay, Vnn=self._Vnn, Vnnn=self._Vnnn),
        )

    def subsystem_rydberg_population_operators(self) -> tuple[jnp.ndarray, ...]:
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_3_atoms_symmetric(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_3_atoms_inf_V(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        if isinf(float(self._Vnn)):
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_3_atoms_inf_V(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        return (
            H_k_atoms_perfect_blockade(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1),
            H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_3_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, Vnn=0.0, Vnnn=0.0),
        )

    def subsystem_initial_states(self) -> tuple[jnp.ndarray, ...]:
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ),
        )

    def process_fidelity(self, final_states) -> float:
        # Obtained diagonal gate matrix
        if float(self._Vnn) == float(self._Vnnn):
            obtained_gate = jnp.array(
                [
                    1,
                    final_states[0][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[1][0],
                    final_states[2][0],
                ]
            )
        elif isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            obtained_gate = jnp.array(
                [
                    1,
                    final_states[0][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[0][0],
                    final_states[0][0] ** 2,
                    final_states[1][0],
                    final_states[2][0],
                ]
            )
        else:
            obtained_gate = jnp.array(
                [
                    1,
                    final_states[0][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[0][0],
                    final_states[2][0],
                    final_states[1][0],
                    final_states[3][0],
                ]
            )

        # Targeted diagonal gate matrix
        p = jnp.angle(obtained_gate[1]) if self._phi is None else self._phi
        t = jnp.angle(obtained_gate[3]) - 2 * p if self._theta is None else self._theta
        e = (
            jnp.angle(obtained_gate[5]) - 2 * p
            if self._theta_prime is None
            else self._theta_prime
        )
        l = (
            jnp.angle(obtained_gate[7]) - 3 * p - 2 * t - e
            if self._lamb is None
            else self._lamb
        )

        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * (3 * p + 2 * t + e + l)),
            ]
        )

        return (
            jnp.abs(jnp.vdot(targeted_gate, obtained_gate)) ** 2
            / len(targeted_gate) ** 2
        )

    def rydberg_time(self, expectation_values) -> float:
        if float(self._Vnn) == float(self._Vnnn):
            return (1 / 8) * float(
                jnp.squeeze(
                    3 * expectation_values[0]
                    + 3 * expectation_values[1]
                    + expectation_values[2]
                )
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (1 / 8) * float(
                jnp.squeeze(
                    5 * expectation_values[0]
                    + 2 * expectation_values[1]
                    + expectation_values[2]
                )
            )
        else:
            return (1 / 8) * float(
                jnp.squeeze(
                    3 * expectation_values[0]
                    + 2 * expectation_values[1]
                    + expectation_values[2]
                    + expectation_values[3]
                )
            )
