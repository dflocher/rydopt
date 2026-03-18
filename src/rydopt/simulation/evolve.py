from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rydopt.pulses.mapped_pulse import MappedPulseAnsatz
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.protocols import Evolvable, PulseAnsatzLike
from rydopt.types import HamiltonianFunction, PulseParams


def evolve(gate: Evolvable, pulse: PulseAnsatzLike | MappedPulseAnsatz, params: PulseParams, tol: float = 1e-7) -> (
        tuple)[jax.Array, ...]:
    r"""The function performs the time evolution of all initial states :math:`|\psi_i(0)\rangle` (specified in the gate
    object), under the pulse Hamiltonian :math:`H`.

    .. math::

        |\psi_i(T)\rangle = U(T)|\psi_i(0)\rangle = \mathcal{T} e^{-\frac{i}{\hbar} \int_0^T H(t)dt}  |\psi_i(0)\rangle

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
        >>> params = (7.61140652, [0.07842706], [1.80300902, -0.61792703], [])
        >>> time_evolved_basis_states = ro.simulation.evolve(gate, pulse, params)

    Args:
        gate: RydOpt Gate object.
        pulse: RydOpt PulseAnsatz object.
        params: Pulse parameters.
        tol: Precision of the ODE solver, default is 1e-7.

    Returns:
        Time-evolved basis states :math:`\{|\psi_i(T)\rangle\}`.

    """
    # When we import diffrax, at least one jnp array is allocated (see optimistix/_misc.py, line 138). Thus,
    # if we change the default device after we have imported diffrax, some memory is allocated on the
    # wrong device. Hence, we defer the import of diffrax to the latest time possible.
    import diffrax

    # If we are on a GPU, dispatch to a GPU-optimized evolve. On GPUs, it is more efficient to solve one
    # large differential equation instead of many small ones because it reduced overheads with kernels.
    if jax.devices()[0].platform == "gpu":
        return _evolve_optimized_for_gpus(gate, pulse, params, tol)

    # Collect initial states and pad them to a common dimension so we can stack
    initial_states = gate.initial_basis_states()

    dims = tuple(len(psi) for psi in initial_states)
    max_dim = max(dims)

    initial_states_padded = jnp.stack([jnp.pad(psi, (0, max_dim - dim)) for psi, dim in zip(initial_states, dims)])

    # Schrödinger equation for the basis states. The Hamiltonian is chosen via lax.switch
    # based on the index of the basis state, with padding to max_dim × max_dim.
    def apply_hamiltonian(
        t: jax.Array | float,
        params: PulseParams,
        psi: jax.Array,
        hamiltonian: HamiltonianFunction,
        dim: int,
    ) -> jax.Array:
        if isinstance(pulse, MappedPulseAnsatz):
            values = pulse.evaluate_pulse_functions_for_gate(t, params, gate)
        else:
            values = pulse.evaluate_pulse_functions(t, params)
        dpsi_small = -1j * hamiltonian(*values) @ psi[:dim]
        return jnp.pad(dpsi_small, (0, psi.shape[0] - dim))

    branches = tuple(
        partial(apply_hamiltonian, hamiltonian=h, dim=d)
        for h, d in zip(gate.hamiltonian_functions_for_basis_states(), dims)
    )

    def schroedinger_eq(t: jax.Array | float, psi: jax.Array, args: tuple[PulseParams, int]) -> jax.Array:
        params, idx = args
        return jax.lax.switch(idx, branches, t, params, psi)

    # Propagator
    term = diffrax.ODETerm(schroedinger_eq)  # type: ignore[arg-type]
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=0.1 * tol, atol=0.1 * tol)
    saveat = diffrax.SaveAt(t1=True)

    def propagate(args: tuple[jax.Array, int]) -> jax.Array:
        psi_initial, idx = args
        duration = pulse.generate_duration(gate, params) if isinstance(pulse, MappedPulseAnsatz) else (params)[0]

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=duration,
            dt0=None,
            y0=psi_initial,
            args=(params, idx),
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=200_000,
        )
        return sol.ys[0]

    # Run the propagator for each basis state
    final_states_padded = jax.lax.map(
        propagate,
        (initial_states_padded, jnp.arange(len(branches))),
    )

    # Remove padding and return original state sizes
    return tuple(s[:d] for s, d in zip(final_states_padded, dims))


def _evolve_optimized_for_gpus(
    gate: Evolvable, pulse: PulseAnsatzLike, params: PulseParams, tol: float = 1e-7
) -> tuple[jax.Array, ...]:
    # When we import diffrax, at least one jnp array is allocated (see optimistix/_misc.py, line 138). Thus,
    # if we change the default device after we have imported diffrax, some memory is allocated on the
    # wrong device. Hence, we defer the import of diffrax to the latest time possible.
    import diffrax

    def schroedinger_eq(
        t: jax.Array | float,
        psi_tuple: tuple[jax.Array, ...],
        _: object,
    ) -> tuple[jax.Array, ...]:
        if isinstance(pulse, MappedPulseAnsatz):
            values = pulse.evaluate_pulse_functions_for_gate(t, params, gate)
        else:
            values = pulse.evaluate_pulse_functions(t, params)
        return tuple(
            -1j * (h(*values) @ psi) for h, psi in zip(gate.hamiltonian_functions_for_basis_states(), psi_tuple)
        )

    solver = diffrax.Dopri8()
    stepsize_controller = diffrax.PIDController(rtol=0.1 * tol, atol=0.1 * tol)
    saveat = diffrax.SaveAt(t1=True)
    term = diffrax.ODETerm(schroedinger_eq)  # type: ignore[arg-type]

    duration = pulse.generate_duration(gate, params) if isinstance(pulse, MappedPulseAnsatz) else (params)[0]
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=duration,
        dt0=None,
        y0=gate.initial_basis_states(),
        args=None,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=200_000,
    )

    return tuple(psi_t1[0] for psi_t1 in sol.ys)
