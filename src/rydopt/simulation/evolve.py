import jax.numpy as jnp
from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
import jax
from functools import partial


def evolve(gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7):
    # When we import diffrax, at least one jnp array is allocated (see optimistix/_misc.py, line 138). Thus,
    # if we change the default device after we have imported diffrax, some memory is allocated on the
    # wrong device. Hence, we defer the import of diffrax to the latest time possible.
    import diffrax

    # If we are on a GPU, dispatch to a GPU-optimized evolve. On GPUs, it is more efficient to solve one
    # large differential equation instead of many small ones because it reduced overheads with kernels.
    if jax.devices()[0].platform == "gpu":
        return _evolve_optimized_for_gpus(gate, pulse, params, tol)

    duration, detuning_params, phase_params, rabi_params = params

    detuning_params = jnp.asarray(detuning_params)
    phase_params = jnp.asarray(phase_params)
    rabi_params = jnp.asarray(rabi_params)

    # Collect initial states and pad them to a common dimension so we can stack
    initial_states = tuple(jnp.asarray(psi) for psi in gate.initial_states())

    dims = tuple(len(psi) for psi in initial_states)
    max_dim = max(dims)

    initial_states_padded = jnp.stack(
        [jnp.pad(psi, (0, max_dim - dim)) for psi, dim in zip(initial_states, dims)]
    )

    # Schrödinger equation for the subsystems. The subsystem Hamiltonian is chosen via lax.switch
    # based on the index of the subsystem, with padding to max_dim × max_dim.
    def apply_hamiltonian(detuning, phase, rabi, psi, hamiltonian, dim):
        dpsi_small = -1j * hamiltonian(detuning, phase, rabi) @ psi[:dim]
        return jnp.pad(dpsi_small, (0, psi.shape[0] - dim))

    branches = tuple(
        partial(apply_hamiltonian, hamiltonian=h, dim=d)
        for h, d in zip(gate.subsystem_hamiltonians(), dims)
    )

    def schroedinger_eq(t, psi, args):
        detuning_params, phase_params, rabi_params, idx = args

        detuning = pulse.detuning_ansatz(t, duration, detuning_params)
        phase = pulse.phase_ansatz(t, duration, phase_params)
        rabi = pulse.rabi_ansatz(t, duration, rabi_params)

        return jax.lax.switch(idx, branches, detuning, phase, rabi, psi)

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
            y0=psi_initial,
            args=(detuning_params, phase_params, rabi_params, idx),
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=10_000,
        )
        return sol.ys[0]

    # Run the propagator for each subsystem
    final_states_padded = jax.lax.map(
        propagate,
        (initial_states_padded, jnp.arange(len(branches))),
    )

    # Remove padding and return original per-subsystem sizes
    final_states = tuple(s[:d] for s, d in zip(final_states_padded, dims))

    return final_states


def _evolve_optimized_for_gpus(
    gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7
):
    # When we import diffrax, at least one jnp array is allocated (see optimistix/_misc.py, line 138). Thus,
    # if we change the default device after we have imported diffrax, some memory is allocated on the
    # wrong device. Hence, we defer the import of diffrax to the latest time possible.
    import diffrax

    duration, detuning_params, phase_params, rabi_params = params

    detuning_params = jnp.asarray(detuning_params)
    phase_params = jnp.asarray(phase_params)
    rabi_params = jnp.asarray(rabi_params)

    initial_states = tuple(jnp.asarray(psi) for psi in gate.initial_states())
    subsystem_hamiltonians = tuple(gate.subsystem_hamiltonians())

    def schroedinger_eq(t, psi_tuple, _):
        detuning = pulse.detuning_ansatz(t, duration, detuning_params)
        phase = pulse.phase_ansatz(t, duration, phase_params)
        rabi = pulse.rabi_ansatz(t, duration, rabi_params)
        return tuple(
            -1j * (h(detuning, phase, rabi) @ psi)
            for h, psi in zip(subsystem_hamiltonians, psi_tuple)
        )

    solver = diffrax.Dopri8()
    stepsize_controller = diffrax.PIDController(rtol=0.1 * tol, atol=0.1 * tol)
    saveat = diffrax.SaveAt(t1=True)
    term = diffrax.ODETerm(schroedinger_eq)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=duration,
        dt0=None,
        y0=initial_states,
        args=None,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=10_000,
    )

    final_states = tuple(psi_t1[0] for psi_t1 in sol.ys)

    return final_states
