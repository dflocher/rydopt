from abc import ABC, abstractmethod


class Gate(ABC):
    @abstractmethod
    def dim(self): ...

    @abstractmethod
    def subsystem_hamiltonians(self): ...

    @abstractmethod
    def initial_states(self): ...

    @abstractmethod
    def process_fidelity(self, final_states): ...

    @abstractmethod
    def rydberg_time(self, ryd_times_subsystems): ...

    def set_decay(self, decay): ...
