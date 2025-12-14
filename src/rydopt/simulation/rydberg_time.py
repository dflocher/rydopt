from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rydopt.protocols import RydbergSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import PulseParams


def rydberg_time(gate: RydbergSystem, pulse: PulseAnsatz, params: PulseParams, tol: float = 1e-7) -> jnp.ndarray:
    r"""The function determines the total time spent in Rydberg states during a gate pulse:

    .. math::

        \Omega_0 T_R = \Omega_0 \int_0^T \sum_{i=1}^{N} \bra{+}^{\otimes N}U(t)^{\dagger}
        |r_i\rangle\!\langle r_i|  U(t)\ket{+}^{\otimes N} dt .

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.TwoQubitGate(
        ...     phi=None,
        ...     theta=np.pi,
        ...     Vnn=float("inf"),
        ...     decay=0,
        ... )
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> params = (7.61140652, [-0.07842706], [1.80300902, -0.61792703], [])
        >>> time_in_rydberg_state = ro.simulation.rydberg_time(gate, pulse, params)

    Args:
        gate: RydOpt Gate object.
        pulse: RydOpt PulseAnsatz object.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-7.

    Returns:
        Total Rydberg time :math:`\Omega_0 T_R`.

    """
    # When we import diffrax, at least one jnp array is allocated (see optimistix/_misc.py, line 138). Thus,
    # if we change the default device after we have imported diffrax, some memory is allocated on the
    # wrong device. Hence, we defer the import of diffrax to the latest time possible.
    import diffrax

    # Collect initial states and pad them to a common dimension so we can stack
    initial_states = gate.initial_basis_states()

    dims = tuple(len(psi) for psi in initial_states)
    max_dim = max(dims)

    initial_states_padded = jnp.stack([jnp.pad(psi, (0, max_dim - dim)) for psi, dim in zip(initial_states, dims)])

    # Schrödinger equation for the basis states. The Hamiltonian is chosen via lax.switch
    # based on the index of the basis state, with padding to max_dim × max_dim.
    def apply_hamiltonian(t, params, y, hamiltonian, rydberg_operator, dim):
        values = pulse.evaluate_pulse_functions(t, params)
        psi, _expectation = y
        psi_small = psi[:dim]
        dpsi_small = -1j * hamiltonian(*values) @ psi_small
        instantaneous_rydberg_population = jnp.vdot(psi_small, rydberg_operator @ psi_small)
        return (
            jnp.pad(dpsi_small, (0, psi.shape[0] - dim)),
            instantaneous_rydberg_population,
        )

    branches = tuple(
        partial(apply_hamiltonian, hamiltonian=h, rydberg_operator=r, dim=d)
        for h, r, d in zip(
            gate.hamiltonian_functions_for_basis_states(),
            gate.rydberg_population_operators_for_basis_states(),
            dims,
        )
    )

    def schroedinger_eq(t, y, args):
        params, idx = args
        return jax.lax.switch(idx, branches, t, params, y)

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
            t1=params[0],
            dt0=None,
            y0=(psi_initial, jnp.array(0.0, dtype=psi_initial.dtype)),
            args=(params, idx),
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=100_000,
        )
        return jnp.real(sol.ys[1])

    # Run the propagator for each basis state
    expectation_values = jax.lax.map(
        propagate,
        (initial_states_padded, jnp.arange(len(branches))),
    )

    return gate.rydberg_time(tuple(expectation_values))
