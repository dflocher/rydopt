import jax.numpy as jnp
import numpy as np

import rydopt as ro


def test_two_photon_effective_controls_for_constant_pulses() -> None:
    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const,
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const,
    )

    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(1, 1, 1),
    )

    params = (5.0, [2.0, 3.0], [0.1, 0.2], [4.0, 6.0])
    detuning, phase, rabi = pulse.evaluate_pulse_functions(1.25, params)
    assert np.allclose(detuning, 2.5)
    assert np.allclose(phase, 0.3)
    assert np.allclose(rabi, 6.0)

    times = jnp.linspace(0.0, params[0], 7)
    detuning_t, phase_t, rabi_t = pulse.evaluate_pulse_functions(times, params)
    assert detuning_t.shape == times.shape
    assert phase_t.shape == times.shape
    assert rabi_t.shape == times.shape
    assert np.allclose(detuning_t, 2.5)
    assert np.allclose(phase_t, 0.3)
    assert np.allclose(rabi_t, 6.0)


def test_two_photon_uses_packed_lower_upper_parameter_blocks() -> None:
    duration = 7.0
    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const_cos_crab,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=ro.pulses.const,
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const_sin_crab,
    )

    lower_params = (duration, [1.6, 0.4, -0.2], [0.7, -0.3], [2.2])
    upper_params = (duration, [0.5], [0.1], [1.4, 0.2, -0.1])
    packed_params = (
        duration,
        np.array([*lower_params[1], *upper_params[1]]),
        np.array([*lower_params[2], *upper_params[2]]),
        np.array([*lower_params[3], *upper_params[3]]),
    )

    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(len(lower_params[1]), len(lower_params[2]), len(lower_params[3])),
    )

    times = jnp.linspace(0.0, duration, 13)
    detuning, phase, rabi = pulse.evaluate_pulse_functions(times, packed_params)

    lower_detuning, lower_phase, lower_rabi = lower.evaluate_pulse_functions(times, lower_params)
    upper_detuning, upper_phase, upper_rabi = upper.evaluate_pulse_functions(times, upper_params)
    expected_detuning = lower_detuning + upper_detuning + (lower_rabi**2 - upper_rabi**2) / (4.0 * lower_detuning)
    expected_phase = lower_phase + upper_phase
    expected_rabi = lower_rabi * upper_rabi / (2.0 * lower_detuning)

    assert np.allclose(detuning, expected_detuning)
    assert np.allclose(phase, expected_phase)
    assert np.allclose(rabi, expected_rabi)
