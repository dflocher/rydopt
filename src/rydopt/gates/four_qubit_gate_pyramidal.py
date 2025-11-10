# TODO remove the suppression of the error F841 when the class is implemented
# ruff: noqa: F841

from functools import partial
import jax.numpy as jnp
from math import isinf
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_2LS_1,
    H_2LS_sqrt2,
    H_2LS_sqrt3,
    H_2LS_sqrt4,
    H_3LS_Vnn,
    H_3LS_Vnnn,
    H_4LS,
    H_4LS_Vnnn,
    H_5LS,
    H_6LS,
    H_8LS,
)


class FourQubitGatePyramidal(Gate):
    def __init__(self, phi, theta, eps, lamb, delta, kappa, Vnn, Vnnn, decay):
        self._phi = phi
        self._theta = theta
        self._eps = eps
        self._lamb = lamb
        self._delta = delta
        self._kappa = kappa
        self._Vnn = Vnn
        self._Vnnn = Vnnn
        self._decay = decay

    def subsystem_hamiltonians(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                partial(H_2LS_1, decay=self._decay),
                partial(H_2LS_sqrt2, decay=self._decay),
                partial(H_2LS_sqrt3, decay=self._decay),
                partial(H_2LS_sqrt4, decay=self._decay),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                partial(H_2LS_1, decay=self._decay),
                partial(H_2LS_sqrt2, decay=self._decay),
                partial(H_4LS, decay=self._decay, Vnnn=self._Vnnn),
                partial(H_5LS, decay=self._decay, Vnnn=self._Vnnn),
            )
        if isinf(float(self._Vnn)):
            return (
                partial(H_2LS_1, decay=self._decay),
                partial(H_2LS_sqrt2, decay=self._decay),
                partial(H_3LS_Vnnn, decay=self._decay, V=self._Vnnn),
                partial(H_4LS, decay=self._decay, Vnnn=self._Vnnn),
                partial(H_4LS_Vnnn, decay=self._decay, Vnnn=self._Vnnn),
                partial(H_5LS, decay=self._decay, Vnnn=self._Vnnn),
            )
        return (
            partial(H_2LS_1, decay=self._decay),
            partial(H_3LS_Vnn, decay=self._decay, V=self._Vnn),
            partial(H_3LS_Vnnn, decay=self._decay, V=self._Vnnn),
            partial(H_6LS, decay=self._decay, Vnn=self._Vnn, Vnnn=self._Vnnn),
            partial(H_4LS_Vnnn, decay=self._decay, Vnnn=self._Vnnn),
            partial(H_8LS, decay=self._decay, Vnn=self._Vnn, Vnnn=self._Vnnn),
        )

    def initial_states(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            )
        return (
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
            ),
            jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [
                    1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ]
            ),
        )

    def target_states(self):
        p = 0.0 if self._phi is None else self._phi
        t = 0.0 if self._theta is None else self._theta
        e = 0.0 if self._eps is None else self._eps
        l = 0.0 if self._lamb is None else self._lamb
        d = 0.0 if self._delta is None else self._delta
        k = 0.0 if self._kappa is None else self._kappa
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return  # TODO
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return  # TODO
        if isinf(float(self._Vnn)):
            return  # TODO
        return  # TODO

    def multiplicities(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return  # TODO
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return  # TODO
        if isinf(float(self._Vnn)):
            return  # TODO
        return  # TODO

    def phase_eliminator(self):
        free_phi = self._phi is None
        free_theta = self._theta is None
        free_eps = self._eps is None
        free_lamb = self._lamb is None
        free_delta = self._delta is None
        free_kappa = self._kappa is None

        def eliminate_phase(overlaps):
            return  # TODO

        return eliminate_phase
