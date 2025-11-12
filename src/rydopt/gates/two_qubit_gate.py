from functools import partial
import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_2LS,
    H_3LS,
)
from math import isinf


class TwoQubitGate(Gate):
    def __init__(self, phi, theta, Vnn, decay):
        self._phi = phi
        self._theta = theta
        self._Vnn = Vnn
        self._decay = decay

    def subsystem_hamiltonians(self):
        if isinf(float(self._Vnn)):
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_2LS, decay=self._decay, k=2),
            )
        return (
            partial(H_2LS, decay=self._decay, k=1),
            partial(H_3LS, decay=self._decay, V=self._Vnn),
        )

    def initial_states(self):
        if isinf(float(self._Vnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0j]),
        )

    def target_states(self):
        p = 0.0  # if self._phi is None else self._phi
        t = 0.0  # if self._theta is None else self._theta

        if isinf(float(self._Vnn)):
            return (
                jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j]),
            )
        return (
            jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
            jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j, 0.0 + 0j]),
        )

    def multiplicities(self):
        return jnp.array([2, 1])

    def phase_eliminator(self):
        free_phi = self._phi is None
        free_theta = self._theta is None

        def eliminate_phase(overlaps):
            # Make use of phase-degrees-of-freedom: for free_phi, o10 defines phi; for free_theta, o11 defines theta

            o10, o11 = overlaps

            # if free_phi:
            #     alpha = jnp.angle(o10)
            #     o10 *= jnp.exp(-1j * alpha)
            #     o11 *= jnp.exp(-1j * 2 * alpha)
            #
            # if free_theta:
            #     beta = jnp.angle(o11)
            #     o11 *= jnp.exp(-1j * beta)

            if free_phi:
                alpha10 = jnp.angle(o10)
                phi = alpha10
            else:
                phi = self._phi

            if free_theta:
                alpha11 = jnp.angle(o11)
                theta = alpha11 - 2 * phi
            else:
                theta = self._theta

            o10 *= jnp.exp(-1j * phi)
            o11 *= jnp.exp(-1j * (2 * phi + theta))

            return jnp.stack([o10, o11])

        return eliminate_phase
