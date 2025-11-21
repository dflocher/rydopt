from abc import ABC, abstractmethod
from copy import deepcopy


class Gate(ABC):
    def __init__(self, decay):
        self._decay = decay

    def copy(self):
        return deepcopy(self)

    def get_decay(self):
        return self._decay

    def set_decay(self, decay):
        self._decay = decay

    @abstractmethod
    def dim(self): ...

    @abstractmethod
    def subsystem_hamiltonians(self): ...

    @abstractmethod
    def subsystem_rydberg_population_operators(self): ...

    @abstractmethod
    def initial_states(self): ...

    @abstractmethod
    def process_fidelity(self, final_states): ...

    @abstractmethod
    def rydberg_time(self, expectation_values): ...
