from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import jax.numpy as jnp

from rydopt.types import HamiltonianFunction


class EvolvableGate(Protocol):
    """Minimal interface needed to run time evolution.

    Used by :func:`rydopt.simulation.evolve`.

    """

    def initial_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension are
        provided.

        Returns:
            Tuple of arrays.

        """
        ...

    def hamiltonians_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        r"""The full gate Hamiltonian can be split into distinct blocks that describe the time evolution
        of basis states. The number of blocks and their dimensionality depends on the interaction strengths.

        Returns:
            Tuple of Hamiltonian functions.

        """
        ...


@runtime_checkable
class OptimizableGate(EvolvableGate, Protocol):
    """Interface for gates that can be optimized for process fidelity.

    Used by :func:`rydopt.optimization.optimize`.
    """

    def process_fidelity(self, final_basis_states: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        r"""Given the basis states evolved under the pulse,
        this function calculates the fidelity with respect to the gate's target state, specified by the gate angles
        :math:`\phi, \, \theta, \, \ldots`

        Args:
            final_basis_states: Time-evolved basis states.

        Returns:
            Fidelity with respect to the target state.

        """
        ...

    def dim(self) -> int:
        r"""Hilbert space dimension.

        Returns:
            Dimensionality :math:`2^n`, where :math:`n` is the number of atoms.

        """
        ...


@runtime_checkable
class RydbergObservableGate(EvolvableGate, Protocol):
    """Interface for gates that expose logic to determine the time in the Rydberg state.

    Used by :func:`rydopt.simulation.rydberg_time`.
    """

    def rydberg_population_operators_for_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""For each basis state, the Rydberg population operators count the number of Rydberg excitations on
        the diagonal.

        Returns:
            Tuple of operators.

        """
        ...

    def rydberg_time(self, expectation_values_of_basis_states) -> jnp.ndarray:
        r"""Given the expectation values of Rydberg populations for each basis state, integrated over the full
        pulse, this function calculates the average time spent in Rydberg states during the gate.

        Args:
            expectation_values_of_basis_states: Expected Rydberg times for each basis state.

        Returns:
            Averaged Rydberg time :math:`T_R`.

        """
        ...


@runtime_checkable
class WithDecayGate(Protocol):
    """Interface for creating a gate with a new decay strength.

    Used by :func:`rydopt.characterization.analyze_gate`.

    """

    def with_decay(self, decay: float) -> Self:
        r"""Creates a copy of the gate with a new decay strength.

        Args:
            decay: New decay strength :math:`\gamma/\Omega_0`.

        Returns:
            A copy of the gate object with the new decay strength.

        """
        ...
