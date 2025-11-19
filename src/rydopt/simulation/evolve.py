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
    tol,
) -> jnp.ndarray:
    term = diffrax.ODETerm(eq)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(
        rtol=0.1 * tol, atol=0.1 * tol
    )  # diffrax.ConstantStepSize()

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
        adjoint=diffrax.DirectAdjoint(),
    )

    return sol.ys[0]


def evolve(gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7):
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
            tol,
        )
        for hamiltonian, psi_initial in zip(
            gate.subsystem_hamiltonians(), gate.initial_states()
        )
    )

    return final_states


# TODO: implement the function. The operators whose expectation values must be calculated are the subsystem Hamiltonians
# TODO: with Delta=1, Omega=Vnn=Vnnn=decay=0  (this counts the number of Rydberg excitations)
def evolve_TR(gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7):
    ryd_times_subsystems = tuple(
        0.0
        for hamiltonian, psi_initial in zip(
            gate.subsystem_hamiltonians(), gate.initial_states()
        )
    )
    return ryd_times_subsystems
