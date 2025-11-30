from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import jax.numpy as jnp

from rydopt.types import HamiltonianFunction


class Gate(ABC):
    r"""Abstract base class that describes Rydberg gates. A Rydberg gate object consists of

    (i) The physical setting (number of atoms, Rydberg interaction strengths between the atoms, Rydberg decay rate).

    (ii) The target gate angles.

    A specific implementation of this base class specifies the number of atoms and their geometric arrangement
    (see examples below).
    The reason for this is that the Hamiltonian describing a gate pulse on a group of atoms is block-diagonal.
    The Rydberg interaction strengths determine how many equivalent blocks there are and what their dimensionality is.

    Args:
        decay: Rydberg decay strength :math:`\gamma/\Omega_0`.

    """

    def __init__(self, decay: float):
        self._decay = decay

    def copy(self) -> Gate:
        r"""Returns:
        A copy of the gate object.

        """
        return deepcopy(self)

    def get_decay(self) -> float:
        r"""Returns:
        Decay strength :math:`\gamma/\Omega_0`.

        """
        return self._decay

    def set_decay(self, decay: float) -> None:
        r"""Args:
        decay: New decay strength :math:`\gamma/\Omega_0`.

        """
        self._decay = decay

    @abstractmethod
    def dim(self) -> int:
        r"""Returns:
        Dimensionality :math:`2^n`, where :math:`n` is the number of atoms.

        """
        ...

    @abstractmethod
    def get_gate_angles(self) -> tuple[float | None, ...]:
        r"""Returns:
        Gate angles.

        """
        ...

    @abstractmethod
    def get_interactions(self) -> tuple[float | None, ...] | float | None:
        r"""Returns:
        Interactions between the atoms.

        """
        ...

    @abstractmethod
    def hamiltonians_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        r"""The full gate Hamiltonian can be split into distinct blocks that describe the time evolution
        of basis states. The number of blocks and their dimensionality depends on the interaction strengths.

        Returns:
            Tuple of Hamiltonian functions.

        """
        ...

    @abstractmethod
    def rydberg_population_operators_for_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""For each basis state, the Rydberg population operators count the number of Rydberg excitations on
        the diagonal.

        Returns:
            Tuple of operators.

        """
        ...

    @abstractmethod
    def initial_basis_states(self) -> tuple[jnp.ndarray, ...]:
        r"""The initial basis states :math:`(1, 0, ...)` of appropriate dimension are
        provided.

        Returns:
            Tuple of arrays.

        """
        ...

    @abstractmethod
    def process_fidelity(self, final_basis_states) -> jnp.ndarray:
        r"""Given the basis states evolved under the pulse,
        this function calculates the fidelity with respect to the gate's target state, specified by the gate angles
        :math:`\phi, \, \theta, \, \ldots`

        Args:
            final_basis_states: Time-evolved basis states.

        Returns:
            Fidelity with respect to the target state.

        """
        ...

    @abstractmethod
    def rydberg_time(self, expectation_values_of_basis_states) -> jnp.ndarray:
        r"""Given the expectation values of Rydberg populations for each basis state, integrated over the full
        pulse, this function calculates the average time spent in Rydberg states during the gate.

        Args:
            expectation_values_of_basis_states: Expected Rydberg times for each basis state.

        Returns:
            Averaged Rydberg time :math:`T_R`.

        """
        ...
