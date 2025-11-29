from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import ParamsTuple


def rydberg_time(gate: Gate, pulse: PulseAnsatz, params: ParamsTuple, tol: float = 1e-7) -> float:
    r"""The function determines the total time spent in Rydberg states during a gate pulse:

    .. math::

        \Omega_0 T_R = \Omega_0 \int_0^T \sum_{i=1}^{N} \bra{+}^{\otimes N}U(t)^{\dagger}
        |r_i\rangle\!\langle r_i|  U(t)\ket{+}^{\otimes N} dt .

    Args:
        gate: rydopt Gate object.
        pulse: rydopt PulseAnsatz object.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-7.

    Returns:
        Total Rydberg time :math:`\Omega_0 T_R`.

    """
    # When we import diffrax, at least one jnp array is allocated (see optimistix/_misc.py, line 138). Thus,
    # if we change the default device after we have imported diffrax, some memory is allocated on the
    # wrong device. Hence, we defer the import of diffrax to the latest time possible.
    import diffrax

    duration, detuning_params, phase_params, rabi_params = params

    detuning_params = jnp.asarray(detuning_params)
    phase_params = jnp.asarray(phase_params)
    rabi_params = jnp.asarray(rabi_params)

    # Collect initial states and pad them to a common dimension so we can stack
    initial_states = gate.subsystem_initial_states()

    dims = tuple(len(psi) for psi in initial_states)
    max_dim = max(dims)

    initial_states_padded = jnp.stack([jnp.pad(psi, (0, max_dim - dim)) for psi, dim in zip(initial_states, dims)])

    # Schrödinger equation for the subsystems. The subsystem Hamiltonian is chosen via lax.switch
    # based on the index of the subsystem, with padding to max_dim × max_dim.
    def apply_hamiltonian(detuning, phase, rabi, y, hamiltonian, rydberg_operator, dim):
        psi, _expectation = y
        psi_small = psi[:dim]
        dpsi_small = -1j * hamiltonian(detuning, phase, rabi) @ psi_small
        instantaneous_rydberg_population = jnp.vdot(psi_small, rydberg_operator @ psi_small)
        return (
            jnp.pad(dpsi_small, (0, psi.shape[0] - dim)),
            instantaneous_rydberg_population,
        )

    branches = tuple(
        partial(apply_hamiltonian, hamiltonian=h, rydberg_operator=r, dim=d)
        for h, r, d in zip(
            gate.subsystem_hamiltonians(),
            gate.subsystem_rydberg_population_operators(),
            dims,
        )
    )

    def schroedinger_eq(t, y, args):
        detuning_params, phase_params, rabi_params, idx = args

        detuning = pulse.detuning_ansatz(t, duration, detuning_params)
        phase = pulse.phase_ansatz(t, duration, phase_params)
        rabi = pulse.rabi_ansatz(t, duration, rabi_params)

        return jax.lax.switch(idx, branches, detuning, phase, rabi, y)

    # Propagator
    term = diffrax.ODETerm(schroedinger_eq)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=0.1 * tol, atol=0.1 * tol)
    saveat = diffrax.SaveAt(t1=True)

    def propagate(args):
        psi_initial, idx = args
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=duration,
            dt0=None,
            y0=(psi_initial, jnp.array(0.0, dtype=psi_initial.dtype)),
            args=(detuning_params, phase_params, rabi_params, idx),
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=100_000,
        )
        return jnp.real(sol.ys[1])

    # Run the propagator for each subsystem
    expectation_values = jax.lax.map(
        propagate,
        (initial_states_padded, jnp.arange(len(branches))),
    )

    return gate.rydberg_time(tuple(expectation_values))
