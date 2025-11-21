import jax.numpy as jnp
import diffrax
from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz


def _propagate(
    psi_initial: jnp.ndarray,
    eq,
    duration: float,
    detuning_params,
    phase_params,
    rabi_params,
    tol,
) -> jnp.ndarray:
    term = diffrax.ODETerm(eq)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=0.1 * tol, atol=0.1 * tol)

    y0 = (psi_initial, jnp.array(0.0, dtype=psi_initial.dtype))

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=duration,
        dt0=None,
        y0=y0,
        args=(detuning_params, phase_params, rabi_params),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=10_000,
    )

    _, expectation_value = sol.ys
    return jnp.real(expectation_value)


def rydberg_time(gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7):
    duration, detuning_params, phase_params, rabi_params = params

    detuning_params = jnp.asarray(detuning_params)
    phase_params = jnp.asarray(phase_params)
    rabi_params = jnp.asarray(rabi_params)

    detuning_fn = lambda t, params: pulse.detuning_ansatz(t, duration, params)  # noqa: E731
    phase_fn = lambda t, params: pulse.phase_ansatz(t, duration, params)  # noqa: E731
    rabi_fn = lambda t, params: pulse.rabi_ansatz(t, duration, params)  # noqa: E731

    def make_schroedinger_eq(hamiltonian, rydberg_operator):
        def eq(t, y, args):
            psi, _ = y

            detuning = detuning_fn(t, args[0])
            phase = phase_fn(t, args[1])
            rabi = rabi_fn(t, args[2])

            dpsi = -1j * (hamiltonian(detuning, phase, rabi) @ psi)
            instantaneous_rydberg_population = jnp.vdot(psi, rydberg_operator @ psi)

            return (dpsi, instantaneous_rydberg_population)

        return eq

    expectation_values = tuple(
        _propagate(
            psi_initial,
            make_schroedinger_eq(hamiltonian, rydberg_operator),
            duration,
            detuning_params,
            phase_params,
            rabi_params,
            tol,
        )
        for hamiltonian, rydberg_operator, psi_initial in zip(
            gate.subsystem_hamiltonians(),
            gate.subsystem_rydberg_population_operators(),
            gate.initial_states(),
        )
    )

    return gate.rydberg_time(expectation_values)
