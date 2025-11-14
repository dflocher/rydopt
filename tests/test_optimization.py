import numpy as np
import rydopt as ro
import pytest


@pytest.mark.optimization
def test_adam() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=None,
    )

    # Initial parameters
    initial_params = (7.6, (-0.1,), (1.8, -0.6), ())

    # Run optimization
    params = ro.optimization.adam(gate, pulse, initial_params, num_steps=200)

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, params)
    assert np.allclose(fidelity, 1, rtol=1e-7)


@pytest.mark.optimization
def test_multi_start_adam() -> None:
    tol = 1e-4

    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=2.0, decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const_cos_crab, phase_ansatz=None, rabi_ansatz=None
    )

    # Parameter bounds for choosing random initial parameters
    min_initial_params = (6, (-2, -2, -2), (), ())
    max_initial_params = (9, (2, 2, 2), (), ())

    # Run optimization
    converged_params = ro.optimization.multi_start_adam(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=200,
        num_initializations=10,
        min_converged_initializations=2,
        tol=tol,
        return_all_converged=True,
    )

    # Verify the fidelity
    for params in converged_params:
        fidelity = ro.simulation.process_fidelity(gate, pulse, params, tol=tol)
        assert np.allclose(fidelity, 1, rtol=tol)


@pytest.mark.optimization
def test_fastest() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=None,
    )

    # Parameter bounds for choosing random initial parameters
    min_initial_params = (6, (-2,), (-2, -2), ())
    max_initial_params = (9, (2,), (2, 2), ())

    # Run optimization
    params = ro.optimization.multi_start_adam(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=200,
        num_initializations=40,
        min_converged_initializations=20,
    )

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, params)
    assert np.allclose(fidelity, 1, rtol=1e-7)
