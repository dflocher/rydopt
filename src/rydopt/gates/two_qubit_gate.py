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
        p = 0.0 if self._phi is None else self._phi
        t = 0.0 if self._theta is None else self._theta

        if self._Vnn == float("inf"):
            return (
                jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j]),
            )
        else:
            return (
                jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j, 0.0 + 0j]),
            )

    def multiplicities(self):
        return jnp.array([2, 1])

    def phase_eliminator(self):
        free_phi = self._phi is None
        free_theta = self._theta is None
        m10 = self.multiplicities()[0]

        def eliminate_phase(overlaps):
            # Make use of phase-degree-of-freedoms so that abs(1+2*o10+o11) gets
            # maximized, assuming that |o10|=|o11|=1

            o10, o11 = overlaps

            if free_phi:
                alpha = jnp.angle(o10)
                beta = jnp.angle(o11)
                s = jnp.where(jnp.cos(alpha - 0.5 * beta) >= 0.0, 1.0, -1.0)
                z = s * jnp.exp(-1j * 0.5 * beta)
                o10 *= z
                o11 *= z**2

            base = 1 + m10 * o10

            if free_theta:
                t = jnp.angle(o11) - jnp.angle(base)
                o11 *= jnp.exp(-1j * t)

            return jnp.stack([o10, o11])

        return eliminate_phase
