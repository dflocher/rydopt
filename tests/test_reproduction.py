from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rydopt as ro
from rydopt.pulses.pulse_ansatz import pack_params


@pytest.mark.optimization
def test_reproducing_evered() -> None:
    """Reproduction of the nearly time-optimal gate of https://doi.org/10.1038/s41586-023-06481-y"""
    # We provide every quantity in units of Omega0 = 2pi x 1 MHz,
    # see https://rydopt.readthedocs.io/en/latest/concepts.html#dimensionless-quantities
    omega0 = 2 * np.pi * 1e6

    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=450, decay=0)

    class EveredPhase(ro.pulses.PulseAnsatzFunction):
        def __init__(self) -> None:
            super().__init__(num_params=3)

        def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
            del duration
            a, omega, phi0 = ansatz_params
            return a * jnp.cos(omega * t - phi0)

    evered_phase = EveredPhase()

    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.Const(),
        phase_ansatz=evered_phase,
        rabi_ansatz=ro.pulses.Const(),
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.Const(),
        rabi_ansatz=ro.pulses.Const(),
    )
    pulse = ro.pulses.TwoPhotonPulseAnsatz(lower_transition=lower, upper_transition=upper, decay=0)

    omega_l = 237
    omega_u = 303
    detuning_l = -7.8e3
    detuning_u = -detuning_l - (omega_l**2 - omega_u**2) / (4 * detuning_l)

    initial_params = ro.pulses.PulseParams(
        1.215 * 2 * np.pi / 4.6,
        [detuning_l, detuning_u],
        [2 * np.pi * 0.1122, 1.0431 * 4.6, -0.7318],
        [omega_l, omega_u],
    )

    # Perform optimization with detunings and  Rabi frequencies fixed
    fixed_initial_params = ro.pulses.PulseParams(False, [True, True], [False, False, False], [True, True])
    result = ro.optimization.optimize(gate, pulse, initial_params, fixed_initial_params, num_steps=200, tol=1e-7)

    # Ensure Rydberg population is realized predominantly via the dark state
    params = result.params
    detuning_params = params[1]
    phase_params = params[2]
    detuning_at_beginning = (
        -jax.grad(partial(evered_phase, duration=0.0, ansatz_params=jnp.array(phase_params)))(0.0)
        + detuning_params[0]
        + detuning_params[1]
    )
    assert detuning_at_beginning * detuning_params[0] < 0

    packed_params = pack_params(params)
    # Effective two-photon detuning
    detuning = abs(pulse.evaluate_pulse_functions(0, packed_params)[1].real) - abs(
        pulse.evaluate_pulse_functions(0, packed_params)[0].real
    )
    print(f"Effective two-photon detuning: 2pi x {detuning:.1f} MHz")
    assert np.allclose(detuning, 0, atol=1e-3)

    # Effective two-photon Rabi frequency
    rabi = abs(pulse.evaluate_pulse_functions(0, packed_params)[3].real)
    print(f"Effective two-photon Rabi frequency: 2pi x {rabi:.1f} MHz")
    assert np.allclose(rabi, 4.6, rtol=1e-3)

    # Gate duration
    duration = params[0]
    print(f"Duration (Omega*T / 2pi): {duration * rabi / (2 * np.pi):.3f}")
    assert np.allclose(duration * rabi / (2 * np.pi), 1.215, rtol=1e-3)

    # Infidelity from the finite lifetime of the intermediate state if one starts in |01>,
    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        decay=1 / 110e-9 / omega0,
    )
    final_state = ro.simulation.evolve(gate, pulse, packed_params)
    obtained = jnp.exp(1j * jnp.angle(final_state[0][0])) * final_state[0]
    target = jnp.array([1, 0])
    infidelity = 1 - jnp.abs(jnp.vdot(target, obtained)) ** 2
    print(f"Infidelity due to intermediate state decay if one starts in |01>: {infidelity:.3%}")
    assert np.allclose(infidelity, 0.043e-2, rtol=0.15)

    # Average gate infidelity due to intermediate state decay
    print(
        "Average gate infidelity due to intermediate state decay: "
        f"{1 - ro.simulation.average_gate_fidelity(gate, pulse, packed_params):.3%}"
    )
