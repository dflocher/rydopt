from abc import ABC, abstractmethod


class Gate(ABC):
    def __init__(self, decay):
        self._decay = decay

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
    def rydberg_time(self, ryd_times_subsystems): ...
