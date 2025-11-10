from functools import partial
import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.gates.subspace_hamiltonians import (
    H_2LS,
    H_3LS,
)


class TwoQubitGate(Gate):
    def __init__(self, phi, theta, Vnn, decay):
        self._phi = phi
        self._theta = theta
        self._Vnn = Vnn
        self._decay = decay

    def _build_hamiltonian(self):
        if self._Vnn == float("inf"):
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_2LS, decay=self._decay, k=2),
            )
        else:
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_3LS, decay=self._decay, V=self._Vnn),
            )

    def _build_initial_state(self):
        if self._Vnn == float("inf"):
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
