import jax.numpy as jnp
import diffrax
from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
import jax
from functools import partial


def evolve(gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7):
    duration, detuning_params, phase_params, rabi_params = params

    detuning_params = jnp.asarray(detuning_params)
    phase_params = jnp.asarray(phase_params)
    rabi_params = jnp.asarray(rabi_params)

    # Collect initial states and pad them to a common dimension so we can stack
    initial_states = [jnp.asarray(psi) for psi in gate.initial_states()]

    dims = [len(psi) for psi in initial_states]
    max_dim = max(dims)

    initial_states_padded = jnp.stack(
        [jnp.pad(psi, (0, max_dim - dim)) for psi, dim in zip(initial_states, dims)]
    )

    # Schrödinger equation for the subsystems. The subsystem Hamiltonian is chosen via lax.switch
    # based on the index of the subsystem, with padding to max_dim × max_dim.
    def apply_hamiltonian(detuning, phase, rabi, psi, hamiltonian, dim):
        out_small = -1j * hamiltonian(detuning, phase, rabi) @ psi[:dim]
        pad_width = (0, psi.shape[0] - dim)
        return jnp.pad(out_small, pad_width)

    branches = tuple(
        partial(apply_hamiltonian, hamiltonian=hamiltonian, dim=dim)
        for hamiltonian, dim in zip(gate.subsystem_hamiltonians(), dims)
    )

    def schroedinger_eq(t, psi, args):
        detuning_params, phase_params, rabi_params, idx = args

        detuning = pulse.detuning_ansatz(t, duration, detuning_params)
        phase = pulse.phase_ansatz(t, duration, phase_params)
        rabi = pulse.rabi_ansatz(t, duration, rabi_params)

        return jax.lax.switch(idx, branches, detuning, phase, rabi, psi)

    # Propagator
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=0.1 * tol, atol=0.1 * tol)
    saveat = diffrax.SaveAt(t1=True)
    term = diffrax.ODETerm(schroedinger_eq)

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

    # Call the propagator for each subsystem by looping over the subsystem indices
    final_states_padded = jax.lax.map(
        propagate,
        (initial_states_padded, jnp.arange(len(branches))),
    )

    # Remove padding and return original per-subsystem sizes
    final_states = tuple(final_states_padded[i, :dim] for i, dim in enumerate(dims))

    return final_states


def evolve2(gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7):
    duration, detuning_params, phase_params, rabi_params = params

    detuning_params = jnp.asarray(detuning_params)
    phase_params = jnp.asarray(phase_params)
    rabi_params = jnp.asarray(rabi_params)

    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=0.1 * tol, atol=0.1 * tol)
    saveat = diffrax.SaveAt(t1=True)

    def make_schroedinger_eq(hamiltonian):
        def eq(t, psi, args):
            detuning = pulse.detuning_ansatz(t, duration, args[0])
            phase = pulse.phase_ansatz(t, duration, args[1])
            rabi = pulse.rabi_ansatz(t, duration, args[2])

            return -1j * (hamiltonian(detuning, phase, rabi) @ psi)

        return eq

    def propagate(psi_initial, eq):
        term = diffrax.ODETerm(eq)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=duration,
            dt0=None,
            y0=psi_initial,
            args=(detuning_params, phase_params, rabi_params),
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=10_000,
        )

        return sol.ys[0]

    final_states = tuple(
        propagate(
            psi_initial,
            make_schroedinger_eq(hamiltonian),
        )
        for hamiltonian, psi_initial in zip(
            gate.subsystem_hamiltonians(), gate.initial_states()
        )
    )

    return final_states


def evolve3(gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7):
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

    solver = diffrax.Tsit5()
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

    return sol.ys[0]
