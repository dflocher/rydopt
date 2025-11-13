import jax.numpy as jnp
import diffrax
from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz


def _make_control(fn, duration, default):
    if fn is None:
        return lambda t, _params: default
    return lambda t, params: fn(t, duration, params)


def _propagate(
    psi_initial: jnp.ndarray,
    eq,
    duration: float,
    detuning_params,
    phase_params,
    rabi_params,
) -> jnp.ndarray:
    term = diffrax.ODETerm(eq)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=duration,
        dt0=None,
        y0=psi_initial,
        args=(detuning_params, phase_params, rabi_params),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=10_000,
    )

    return sol.ys[0]


def evolve(gate: Gate, pulse: PulseAnsatz, params: tuple):
    duration, detuning_params, phase_params, rabi_params = params

    detuning_params = jnp.asarray(detuning_params)
    phase_params = jnp.asarray(phase_params)
    rabi_params = jnp.asarray(rabi_params)

    detuning_fn = _make_control(pulse.detuning_ansatz, duration, 0.0)
    phase_fn = _make_control(pulse.phase_ansatz, duration, 0.0)
    rabi_fn = _make_control(pulse.rabi_ansatz, duration, 1.0)

    def make_schroedinger_eq(hamiltonian):
        def eq(t, y, args):
            detuning = detuning_fn(t, args[0])
            phase = phase_fn(t, args[1])
            rabi = rabi_fn(t, args[2])
            return -1j * (hamiltonian(detuning, phase, rabi) @ y)

        return eq

    final_states = tuple(
        _propagate(
            psi_initial,
            make_schroedinger_eq(hamiltonian),
            duration,
            detuning_params,
            phase_params,
            rabi_params,
        )
        for hamiltonian, psi_initial in zip(
            gate.subsystem_hamiltonians(), gate.initial_states()
        )
    )

    return final_states
