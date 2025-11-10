from abc import ABC, abstractmethod


class Gate(ABC):
    def build(self):
        return (
            self._build_hamiltonian(),
            self._build_initial_state(),
            self._build_target_state(),
        )

    @abstractmethod
    def _build_hamiltonian(self): ...

    @abstractmethod
    def _build_initial_state(self): ...

    @abstractmethod
    def _build_target_state(self): ...
