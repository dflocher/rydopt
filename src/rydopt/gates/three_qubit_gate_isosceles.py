from functools import partial
import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.gates.subsystem_hamiltonians import (
    H_k_atoms_perfect_blockade,
    H_2_atoms,
    H_3_atoms_inf_V,
    H_3_atoms_symmetric,
    H_3_atoms,
)
from math import isinf


class ThreeQubitGateIsosceles(Gate):
    def __init__(self, phi, theta, eps, lamb, Vnn, Vnnn, decay):
        super().__init__(decay)
        if (Vnn == Vnnn) and (theta != eps):
            raise ValueError("For Vnn=Vnnn, theta=eps is required")
        if (Vnnn == 0) and (eps != 0.0):
            raise ValueError("For Vnnn=0, eps=0 is required")
        self._phi = phi
        self._theta = theta
        self._eps = eps
        self._lamb = lamb
        self._Vnn = Vnn
        self._Vnnn = Vnnn

    def dim(self):
        return 8

    def get_phi_theta_eps_lamb(self):
        return self._phi, self._theta, self._eps, self._lamb

    def get_Vnn_Vnnn(self):
        return self._Vnn, self._Vnnn

    def subsystem_hamiltonians(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=3),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_2_atoms, decay=self._decay, V=self._Vnn),
                partial(H_3_atoms_symmetric, decay=self._decay, V=self._Vnn),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_3_atoms_inf_V, decay=self._decay, V=self._Vnnn),
            )
        if isinf(float(self._Vnn)):
            return (
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
                partial(H_k_atoms_perfect_blockade, decay=self._decay, k=2),
                partial(H_2_atoms, decay=self._decay, V=self._Vnnn),
                partial(H_3_atoms_inf_V, decay=self._decay, V=self._Vnnn),
            )
        return (
            partial(H_k_atoms_perfect_blockade, decay=self._decay, k=1),
            partial(H_2_atoms, decay=self._decay, V=self._Vnn),
            partial(H_2_atoms, decay=self._decay, V=self._Vnnn),
            partial(H_3_atoms, decay=self._decay, Vnn=self._Vnn, Vnnn=self._Vnnn),
        )

    def subsystem_rydberg_population_operators(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_3_atoms_symmetric(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_3_atoms_inf_V(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        if isinf(float(self._Vnn)):
            return (
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_k_atoms_perfect_blockade(
                    Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1
                ),
                H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
                H_3_atoms_inf_V(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            )
        return (
            H_k_atoms_perfect_blockade(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, k=1),
            H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_2_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, V=0.0),
            H_3_atoms(Delta=1.0, Phi=0.0, Omega=0.0, decay=0.0, Vnn=0.0, Vnnn=0.0),
        )

    def initial_states(self):
        if isinf(float(self._Vnn)) and isinf(float(self._Vnnn)):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
            )
        if float(self._Vnn) == float(self._Vnnn):
            return (
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
                jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
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
                    final_states[0][0] ** 2,
                    final_states[1][0],
                    final_states[2][0],
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
                    final_states[2][0],
                    final_states[1][0],
                    final_states[3][0],
                ]
            )

        # Targeted diagonal gate matrix
        p = jnp.angle(obtained_gate[1]) if self._phi is None else self._phi
        t = jnp.angle(obtained_gate[3]) - 2 * p if self._theta is None else self._theta
        e = jnp.angle(obtained_gate[5]) - 2 * p if self._eps is None else self._eps
        l = (
            jnp.angle(obtained_gate[7]) - 3 * p - 2 * t - e
            if self._lamb is None
            else self._lamb
        )

        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + e)),
                jnp.exp(1j * (2 * p + t)),
                jnp.exp(1j * (3 * p + 2 * t + e + l)),
            ]
        )

        return (
            jnp.abs(jnp.vdot(targeted_gate, obtained_gate)) ** 2
            / len(targeted_gate) ** 2
        )

    def rydberg_time(self, ryd_times_subsystems):
        if float(self._Vnn) == float(self._Vnnn):
            return (1 / 8) * (
                3 * ryd_times_subsystems[0]
                + 3 * ryd_times_subsystems[1]
                + ryd_times_subsystems[2]
            )
        if isinf(float(self._Vnn)) and float(self._Vnnn) == 0.0:
            return (1 / 8) * (
                5 * ryd_times_subsystems[0]
                + 2 * ryd_times_subsystems[1]
                + ryd_times_subsystems[2]
            )
        else:
            return (1 / 8) * (
                3 * ryd_times_subsystems[0]
                + 2 * ryd_times_subsystems[1]
                + ryd_times_subsystems[2]
                + ryd_times_subsystems[3]
            )
