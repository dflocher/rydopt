from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from typing_extensions import Self

from rydopt.types import HamiltonianFunction


class Evolvable(Protocol):
    """Minimal interface for a system that can be time evolved.

    Used by :func:`rydopt.simulation.evolve`.

    """

    def initial_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension.

        Returns:
            Tuple of initial basis states.

        """
        ...

    def hamiltonian_functions_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        r"""The Hamiltonian under which the initial basis states evolve.

        A separate Hamiltonian function is returned for each initial basis state. In the case of a
        block-diagonal Hamiltonian, this allows for returning only the block that is of relevance
        for a particular basis state.

        Returns:
            Tuple of Hamiltonian functions.

        """
        ...


@runtime_checkable
class GateSystem(Evolvable, Protocol):
    """Interface for :ref:`gates <gates>` that can be optimized for process fidelity.
    The interface is derived from :class:`Evolvable`. Additionally, methods are present for
    calculating fidelities from time-evolved basis states.

    Used by :func:`rydopt.simulation.process_fidelity`, :func:`rydopt.simulation.average_gate_fidelity`,
    :func:`rydopt.optimization.optimize`, :func:`rydopt.characterization.analyze_gate`,
    and :func:`rydopt.characterization.analyze_gate_qutip`.
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
class RydbergSystem(Evolvable, Protocol):
    """Interface for Evolvables that utilize Rydberg states.
    The interface is derived from :class:`Evolvable`. Additionally, methods are present for
    calculating the time spent in Rydberg states during time evolution.

    Used by :func:`rydopt.simulation.rydberg_time`, :func:`rydopt.characterization.analyze_gate`,
    and :func:`rydopt.characterization.analyze_gate_qutip`.
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

    def with_decay(self, decay: float) -> Self:
        r"""Creates a copy of the gate with a new decay strength.

        Args:
            decay: New decay strength :math:`\gamma/\Omega_0`.

        Returns:
            A copy of the gate object with the new decay strength.

        """
        ...
