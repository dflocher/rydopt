from functools import partial
import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_2LS,
    H_3LS,
)


class TwoQubitGate(Gate):
    def __init__(self, phi, theta, Vnn, decay):
        self._phi = phi
        self._theta = theta
        self._Vnn = Vnn
        self._decay = decay

    def subsystem_hamiltonians(self):
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

    def initial_states(self):
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

    def target_states(self):
        if self._Vnn == float("inf"):
            return (
                jnp.array([jnp.exp(1j * self._phi), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * self._phi + self._theta)), 0.0 + 0.0j]),
            )
        else:
            return (
                jnp.array([jnp.exp(1j * self._phi), 0.0 + 0.0j]),
                jnp.array(
                    [jnp.exp(1j * (2 * self._phi + self._theta)), 0.0 + 0.0j, 0.0 + 0j]
                ),
            )

    def multiplicities(self):
        return jnp.array([2, 1])
