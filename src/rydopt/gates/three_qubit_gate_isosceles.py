# TODO remove the suppression of the error F841 when the class is implemented
# ruff: noqa: F841

from functools import partial
import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_2LS,
    H_3LS,
    H_4LS,
    H_4LS_Vnnn,
    H_6LS,
)
from math import isinf


class ThreeQubitGateIsosceles(Gate):
    def __init__(self, phi, theta, eps, lamb, Vnn, Vnnn, decay):
        # TODO: check error cases
        if (Vnnn == Vnnn) and (theta != eps):
            raise IOError("For Vnn=Vnnn, theta=eps is required")
        if (Vnnn == 0) and (eps != 0.0):
            raise IOError("For Vnnn=0, eps=0 is required")
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
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_2LS, decay=self._decay, k=2),
                partial(H_2LS, decay=self._decay, k=3),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_2LS, decay=self._decay, k=2),
                partial(H_4LS, decay=self._decay, Vnnn=self._Vnnn),
            )
        if isinf(float(self._Vnn)):
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_3LS, decay=self._decay, V=self._Vnnn),
                partial(H_2LS, decay=self._decay, k=2),
                partial(H_4LS, decay=self._decay, Vnnn=self._Vnnn),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_3LS, decay=self._decay, V=self._Vnnn),
                partial(H_4LS_Vnnn, decay=self._decay, Vnnn=self._Vnnn),
            )
        return (
            partial(H_2LS, decay=self._decay, k=1),
            partial(H_3LS, decay=self._decay, V=self._Vnnn),
            partial(H_3LS, decay=self._decay, V=self._Vnn),
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
        if float(self._Vnn) == float(self._Vnnn):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
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
        p = 0.0  # if self._phi is None else self._phi
        t = 0.0  # if self._theta is None else self._theta
        e = 0.0  # if self._eps is None else self._eps
        l = 0.0  # if self._lamb is None else self._lamb

        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (3 * p + 2 * t + e + l)), 0.0 + 0.0j]),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j]),
                jnp.array(
                    [
                        jnp.exp(1j * (3 * p + 2 * t + e + l)),
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            )
        if isinf(float(self._Vnn)):
            return (
                jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * p + e)), 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j]),
                jnp.array(
                    [
                        jnp.exp(1j * (3 * p + 2 * t + e + l)),
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
                jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array(
                    [
                        jnp.exp(1j * (3 * p + 2 * t + e + l)),
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            )
        return (
            jnp.array([jnp.exp(1j * p), 0.0 + 0.0j]),
            jnp.array([jnp.exp(1j * (2 * p + e)), 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array([jnp.exp(1j * (2 * p + t)), 0.0 + 0.0j, 0.0 + 0.0j]),
            jnp.array(
                [
                    jnp.exp(1j * (3 * p + 2 * t + e + l)),
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ]
            ),
        )

    def multiplicities(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return jnp.array([3, 3, 1])
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            # TODO: not quite correct: one of the first 4 overlaps must be squared to obtain the correct fidelity
            return jnp.array([4, 2, 1])
        if isinf(float(self._Vnn)):
            return jnp.array([3, 1, 2, 1])
        if float(self._Vnn) == float(self._Vnnn):
            return jnp.array([3, 1, 2, 1])
        return jnp.array([3, 1, 2, 1])

    def phase_eliminator(self):
        free_phi = self._phi is None
        free_theta = self._theta is None
        free_eps = self._eps is None
        free_lamb = self._lamb is None

        def eliminate_phase(overlaps):
            if float(self._Vnn) == float(self._Vnnn):
                o100, o110, o111 = overlaps
                o101 = o110
            elif isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
                o100, o110, o111 = overlaps
                o101 = o100**2
            else:
                o100, o101, o110, o111 = overlaps

            if free_phi:
                alpha100 = jnp.angle(o100)
                phi = alpha100
            else:
                phi = self._phi

            if free_theta:
                alpha110 = jnp.angle(o110)
                theta = alpha110 - 2 * phi
            else:
                theta = self._theta

            if free_eps:
                alpha101 = jnp.angle(o101)
                eps = alpha101 - 2 * phi
            else:
                eps = self._eps

            if free_lamb:
                alpha111 = jnp.angle(o111)
                lamb = alpha111 - 3 * phi - 2 * theta - eps
            else:
                lamb = self._lamb

            o100 *= jnp.exp(-1j * phi)
            o110 *= jnp.exp(-1j * (2 * phi + theta))
            o101 *= jnp.exp(-1j * (2 * phi + eps))
            o111 *= jnp.exp(-1j * (3 * phi + 2 * theta + eps + lamb))

            if (
                float(self._Vnn) == float(self._Vnnn)
                or isinf(float(self._Vnn))
                and float(self._Vnnn) == 0.0
            ):
                return jnp.stack([o100, o110, o111])
            return jnp.stack([o100, o101, o110, o111])

        return eliminate_phase
