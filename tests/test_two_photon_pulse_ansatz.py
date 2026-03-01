import jax.numpy as jnp
import numpy as np
import pytest

import rydopt as ro


def test_effective_controls() -> None:
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


@pytest.mark.optimization
def test_two_photon_cz() -> None:
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=ro.pulses.const,
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const,
    )
    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(1, 2, 1),
        decay=0,
    )

    initial_params = (7.6, [50.0, -50.0], [1.8, -0.6], [10.0, 10.0])  # duration, detuning, phase, rabi

    # Parameters of the upper transition and Rabi frequencies are fixed
    fixed_initial_params = (False, [False, True], [False, False], [True, True])

    result = ro.optimization.optimize(gate, pulse, initial_params, fixed_initial_params, num_steps=200, tol=1e-7)

    ref = (
        7.600019896010689,
        [49.92218101, -50],
        [1.75873066, -0.61830304],
        [10, 10],
    )
    assert all(np.allclose(x, y, rtol=1e-3) for x, y in zip(result.params, ref))


def test_intermediate_state_decay() -> None:
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=ro.pulses.const,
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const,
    )
    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(1, 2, 1),
        decay=0.5,
    )

    def compute_fidelity(intermediate_state_detuning: float) -> float:
        detuning_ref = 49.92218101
        detuning_new = detuning_ref - 50 + intermediate_state_detuning
        params = (
            7.600019896010689,
            [detuning_new, -intermediate_state_detuning],
            [1.75873066, -0.61830304],
            [
                np.sign(detuning_new) * 10 * np.sqrt(np.abs(detuning_new / detuning_ref)),
                10 * np.sqrt(np.abs(detuning_new / detuning_ref)),
            ],
        )
        return float(ro.simulation.process_fidelity(gate, pulse, params))

    infidelity_dark = np.abs(1 - compute_fidelity(50))  # dynamics via the dark state
    infidelity_bright = np.abs(1 - compute_fidelity(-50))  # dynamics via the bright state

    assert infidelity_dark < infidelity_bright
