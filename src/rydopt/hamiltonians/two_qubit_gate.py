from functools import partial
import jax.numpy as jnp
from rydopt.hamiltonians.subspace_hamiltonians import (
    H_2LS,
    H_3LS_Vnn,
)


class TwoQubitGate:
    def __init__(self, theta, Vnn, decay):
        self.theta = theta
        self.Vnn = Vnn
        self.decay = decay

    def build(self):
        return (
            self._build_hamiltonian(),
            self._build_initial_state(),
            self._build_target_state(),
        )

    def _build_hamiltonian(self):
        if self.Vnn == float("inf"):
            return (
                partial(H_2LS, decay=self.decay, k=1),
                partial(H_2LS, decay=self.decay, k=2),
            )
        else:
            return (
                partial(H_2LS, decay=self.decay, k=1),
                partial(H_3LS_Vnn, decay=self.decay, V=self.Vnn),
            )

    def _build_initial_state(self):
        if self.Vnn == float("inf"):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        else:
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0j]),
            )

    def _build_target_state(self):
        # TODO
        return
