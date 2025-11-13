# TODO remove the suppression of the error F841 when the class is implemented
# ruff: noqa: F841

from functools import partial
import jax.numpy as jnp
from math import isinf
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_2LS,
    H_3LS,
    H_4LS,
    H_4LS_Vnnn,
    H_5LS,
    H_6LS,
    H_8LS,
)


class FourQubitGatePyramidal(Gate):
    def __init__(self, phi, theta, eps, lamb, delta, kappa, Vnn, Vnnn, decay):
        # TODO: check error cases
        if (Vnnn == Vnnn) and ((theta != eps) or (lamb != delta)):
            raise IOError("For Vnn=Vnnn, theta=eps and lambda=delta is required")
        if (Vnnn == 0) and ((eps != 0.0) or (delta != 0.0)):
            raise IOError("For Vnnn=0, eps=0 and delta=0 is required")
        self._phi = phi
        self._theta = theta
        self._eps = eps
        self._lamb = lamb
        self._delta = delta
        self._kappa = kappa
        self._Vnn = Vnn
        self._Vnnn = Vnnn
        self._decay = decay

    def dim(self):
        return 16

    def subsystem_hamiltonians(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_2LS, decay=self._decay, k=2),
                partial(H_2LS, decay=self._decay, k=3),
                partial(H_2LS, decay=self._decay, k=4),
            )
        # TODO Vnn = Vnnn
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_2LS, decay=self._decay, k=2),
                partial(H_4LS, decay=self._decay, Vnnn=self._Vnnn),
                partial(H_5LS, decay=self._decay, Vnnn=self._Vnnn),
            )
        if isinf(float(self._Vnn)):
            return (
                partial(H_2LS, decay=self._decay, k=1),
                partial(H_2LS, decay=self._decay, k=2),
                partial(H_3LS, decay=self._decay, V=self._Vnnn),
                partial(H_4LS, decay=self._decay, Vnnn=self._Vnnn),
                partial(H_4LS_Vnnn, decay=self._decay, Vnnn=self._Vnnn),
                partial(H_5LS, decay=self._decay, Vnnn=self._Vnnn),
            )
        return (
            partial(H_2LS, decay=self._decay, k=1),
            partial(H_3LS, decay=self._decay, V=self._Vnn),
            partial(H_3LS, decay=self._decay, V=self._Vnnn),
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

    def process_fidelity(self, final_states):
        # Obtained diagonal gate matrix
        if float(self._Vnn) == float(self._Vnnn):
            obtained_gate = jnp.array(
                [
                    1,
                    final_states[0][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[1][0],
                    final_states[2][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[1][0],
                    final_states[2][0],
                    final_states[1][0],
                    final_states[2][0],
                    final_states[2][0],
                    final_states[3][0],
                ]
            )
        elif isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            obtained_gate = jnp.array(
                [
                    1,
                    final_states[0][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[0][0] ** 2,
                    final_states[2][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[0][0] ** 2,
                    final_states[2][0],
                    final_states[0][0] ** 2,
                    final_states[2][0],
                    final_states[0][0] ** 3,
                    final_states[3][0],
                ]
            )
        else:
            obtained_gate = jnp.array(
                [
                    1,
                    final_states[0][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[2][0],
                    final_states[3][0],
                    final_states[0][0],
                    final_states[1][0],
                    final_states[2][0],
                    final_states[3][0],
                    final_states[2][0],
                    final_states[3][0],
                    final_states[4][0],
                    final_states[5][0],
                ]
            )

        # Targeted diagonal gate matrix
        p = jnp.angle(obtained_gate[1]) if self._phi is None else self._phi
        t = jnp.angle(obtained_gate[3]) - 2 * p if self._theta is None else self._theta
        e = jnp.angle(obtained_gate[6]) - 2 * p if self._eps is None else self._eps
        l = (
            jnp.angle(obtained_gate[7]) - 3 * p - 2 * t - e
            if self._lamb is None
            else self._lamb
        )
        d = (
            jnp.angle(obtained_gate[14]) - 3 * p - 3 * e
            if self._delta is None
            else self._delta
        )
        k = (
            jnp.angle(obtained_gate[15]) - 4 * p - 3 * t - 3 * e - 3 * l - d
            if self._kappa is None
            else self._kappa
        )

        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * (3 * p + 2 * t + e + l)),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * (3 * p + 2 * t + e + l)),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * (3 * p + 2 * t + e + l)),
                jnp.exp(1j * (3 * p + 3 * e + d)),
                jnp.exp(1j * (4 * p + 3 * t + 3 * e + 3 * l + d + k)),
            ]
        )

        return (
            jnp.abs(jnp.vdot(targeted_gate, obtained_gate)) ** 2
            / len(targeted_gate) ** 2
        )
