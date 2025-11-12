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

    def average_gate_fidelity(self, final_states):
        return (self.dim() * self.process_fidelity(final_states) + 1.0) / (
            self.dim() + 1.0
        )
