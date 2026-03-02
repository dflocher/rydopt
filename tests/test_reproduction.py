from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rydopt as ro


@pytest.mark.optimization
def test_reproducing_evered() -> None:
    """Reproduction of the nearly time-optimal gate of https://doi.org/10.1038/s41586-023-06481-y"""
    # We provide every quantity in units of Omega0 = 2pi x 1 MHz,
    # see https://rydopt.readthedocs.io/en/latest/concepts.html#dimensionless-quantities
    omega0 = 2 * np.pi * 1e6

    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=450, decay=0)

    def evered_phase(t: jax.Array | float, _duration: float, ansatz_params: jax.Array) -> jax.Array:
        a, omega, phi0 = ansatz_params
        return a * jnp.cos(omega * t - phi0)

    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=evered_phase,
        rabi_ansatz=ro.pulses.const,
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const,
    )
    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(1, 3, 1),
        decay=0,
    )

    omega_l = 237
    omega_u = 303
    detuning_l = 7.8e3
    detuning_u = -detuning_l - (omega_l**2 - omega_u**2) / (4 * detuning_l)

    initial_params = (
        1.215 * 2 * np.pi / 4.6,
        [detuning_l, detuning_u],
        [2 * np.pi * 0.1122, 1.0431 * 4.6, -0.7318],
        [omega_l, omega_u],
    )  # duration, detuning, phase, rabi

    # Perform optimization with detunings and  Rabi frequencies fixed
    fixed_initial_params = (False, [True, True], [False, False, False], [True, True])
    result = ro.optimization.optimize(gate, pulse, initial_params, fixed_initial_params, num_steps=200, tol=1e-7)

    # Ensure Rydberg population is realized predominantly via the dark state
    detuning_params = np.array(result.params[1])
    phase_params = np.array(result.params[2])
    detuning_at_beginning = (
        jax.grad(partial(evered_phase, _duration=0.0, ansatz_params=jnp.array(phase_params)))(0.0)
        + detuning_params[0]
        + detuning_params[1]
    )
    assert detuning_at_beginning * detuning_params[0] < 0

    # Effective two-photon detuning
    detuning = abs(pulse.evaluate_pulse_functions(0, result.params)[0].real)
    print(f"Effective two-photon detuning: 2pi x {detuning:.1f} MHz")
    assert np.allclose(detuning, 0, atol=1e-3)

    # Effective two-photon Rabi frequency
    rabi = abs(pulse.evaluate_pulse_functions(0, result.params)[2].real)
    print(f"Effective two-photon Rabi frequency: 2pi x {rabi:.1f} MHz")
    assert np.allclose(rabi, 4.6, rtol=1e-3)

    # Gate duration
    duration = result.params[0]
    print(f"Duration (Omega*T / 2pi): {duration * rabi / (2 * np.pi):.3f}")
    assert np.allclose(duration * rabi / (2 * np.pi), 1.215, rtol=1e-3)

    # Infidelity from the finite lifetime of the intermediate state if one starts in |11>,
    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(1, 3, 1),
        decay=1 / 110e-9 / omega0,
    )

    final_state = ro.simulation.evolve(gate, pulse, result.params)
    obtained = jnp.exp(1j * jnp.angle(final_state[0][0])) * final_state[1]
    target = jnp.array([-1, 0, 0])
    infidelity = abs(1 - jnp.abs(jnp.vdot(target, obtained)) ** 2)
    print(f"Infidelity due to intermediate state decay if one starts in |11>: {infidelity:.3%}")
    assert np.allclose(
        infidelity,
        0.043e-2,  # value from Fig. 4 of https://doi.org/10.1038/s41586-023-06481-y
        rtol=1e-1,
    )

    # Infidelity from the finite lifetime of the intermediate state if one starts in |01>,
    obtained = jnp.exp(1j * jnp.angle(final_state[0][0])) * final_state[0]
    target = jnp.array([1, 0])
    infidelity = abs(1 - jnp.abs(jnp.vdot(target, obtained)) ** 2)
    print(f"Infidelity due to intermediate state decay if one starts in |01>: {infidelity:.3%}")
    assert np.allclose(
        infidelity,
        0.019e-2,  # value from Fig. 4 of https://doi.org/10.1038/s41586-023-06481-y
        rtol=1e-1,
    )

    # Average gate infidelity due to intermediate state decay
    print(
        "Average gate infidelity due to intermediate state decay: "
        f"{abs(1 - ro.simulation.average_gate_fidelity(gate, pulse, result.params)):.3%}"
    )
