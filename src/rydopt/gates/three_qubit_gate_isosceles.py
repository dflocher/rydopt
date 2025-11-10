# TODO remove the suppression of the error F841 when the class is implemented
# ruff: noqa: F841

from functools import partial
import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_2LS_1,
    H_2LS_sqrt2,
    H_2LS_sqrt3,
    H_4LS,
    H_3LS_Vnn,
    H_3LS_Vnnn,
    H_6LS,
)
from math import isinf


class ThreeQubitGateIsosceles(Gate):
    def __init__(self, phi, theta, eps, lamb, Vnn, Vnnn, decay):
        self._phi = phi
        self._theta = theta
        self._eps = eps
        self._lamb = lamb
        self._Vnn = Vnn
        self._Vnnn = Vnnn
        self._decay = decay

    def subsystem_hamiltonians(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                partial(H_2LS_1, decay=self._decay),
                partial(H_2LS_sqrt2, decay=self._decay),
                partial(H_2LS_sqrt3, decay=self._decay),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                partial(H_2LS_1, decay=self._decay),
                partial(H_2LS_sqrt2, decay=self._decay),
                partial(H_4LS, decay=self._decay, Vnnn=self._Vnnn),
            )
        if isinf(float(self._Vnn)):
            return (
                partial(H_2LS_1, decay=self._decay),
                partial(H_3LS_Vnnn, decay=self._decay, V=self._Vnnn),
                partial(H_2LS_sqrt2, decay=self._decay),
                partial(H_4LS, decay=self._decay, Vnnn=self._Vnnn),
            )
        return (
            partial(H_2LS_1, decay=self._decay),
            partial(H_3LS_Vnnn, decay=self._decay, V=self._Vnnn),
            partial(H_3LS_Vnn, decay=self._decay, V=self._Vnn),
            partial(H_6LS, decay=self._decay, Vnn=self._Vnn, Vnnn=self._Vnnn),
        )

    def initial_states(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ),
        )

    def target_states(self):
        p = 0.0 if self._phi is None else self._phi
        t = 0.0 if self._theta is None else self._theta
        e = 0.0 if self._eps is None else self._eps
        l = 0.0 if self._lamb is None else self._lamb

        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return  # TODO
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return  # TODO
        if isinf(float(self._Vnn)):
            return  # TODO
        return  # TODO

    def multiplicities(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return jnp.array([3, 3, 1])
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return  # TODO
        if isinf(float(self._Vnn)):
            return jnp.array([3, 1, 2, 1])
        return jnp.array([3, 1, 2, 1])

    def phase_eliminator(self):
        free_phi = self._phi is None
        free_theta = self._theta is None
        free_eps = self._eps is None
        free_lamb = self._lamb is None

        def eliminate_phase(overlaps):
            return  # TODO

        return eliminate_phase
