from abc import ABC, abstractmethod


class Gate(ABC):
    @abstractmethod
    def subsystem_hamiltonians(self): ...

    @abstractmethod
    def initial_states(self): ...

    @abstractmethod
    def target_states(self): ...

    @abstractmethod
    def multiplicities(self): ...

    @abstractmethod
    def phase_eliminator(self): ...
