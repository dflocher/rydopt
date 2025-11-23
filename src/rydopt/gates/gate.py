from abc import ABC, abstractmethod
from copy import deepcopy


class Gate(ABC):
    def __init__(self, decay: float):
        self._decay = decay

    def copy(self) -> "Gate":
        r"""
        Returns:
            A copy of the gate object.
        """
        return deepcopy(self)

    def get_decay(self) -> float:
        r"""
        Returns:
           Decay strength :math:`\gamma/\Omega_0`.
        """
        return self._decay

    def set_decay(self, decay: float) -> None:
        r"""
        Args:
            decay: New decay strength :math:`\gamma/\Omega_0`.
        """
        self._decay = decay

    @abstractmethod
    def dim(self) -> int:
        r"""
        Returns:
            Dimensionality :math:`2^n`, where :math:`n` is the number of qubits.
        """
        ...

    @abstractmethod
    def subsystem_hamiltonians(self) -> tuple:
        r"""
        The full gate Hamiltonian can be split into distinct subsystems.
        The number of subsystem Hamiltonians and their dimensionality depends on the interaction strengths.

        Returns:
            Tuple of subsystem Hamiltonian functions.
        """
        ...

    @abstractmethod
    def subsystem_rydberg_population_operators(self) -> tuple:
        r"""
        For each subsytem Hamiltonian, the Rydberg population operators count the number of Rydberg excitations on the diagonal.

        Returns:
            Tuple of operators.
        """
        ...

    @abstractmethod
    def initial_states(self) -> tuple:
        r"""
        For each subsytem Hamiltonian, the initial states :math:`(1, 0, ...)` of appropriate dimension are provided.

        Returns:
            Tuple of arrays.
        """
        ...

    @abstractmethod
    def process_fidelity(self, final_states) -> float:
        r"""
        Given the states evolved under the pulse subsystem Hamiltonians,
        this function calculates the fidelity with respect to the gate's target state, specified by the gate angles :math:`\phi, \, \theta, \, \ldots`

        Args:
            final_states: states evolved under the subsystem Hamiltonians.

        Returns:
            Fidelity with respect to the target state.
        """
        ...

    @abstractmethod
    def rydberg_time(self, expectation_values) -> float:
        r"""
        Given the expectation values of Rydberg populations for each subsystem Hamiltonian, integrated over the full pulse,
        this function calculates to average time spent in Rydberg states during the gate.

        Args:
            expectation_values: Expected Rydberg times of the subsystems.

        Returns:
            Averaged Rydberg time :math:`T_R`.
        """
        ...
