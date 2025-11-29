import numpy as np
import pytest

import rydopt as ro


@pytest.mark.optimization
def test_adam() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

    # Initial parameters
    initial_params = (7.6, (-0.1,), (1.8, -0.6), ())

    # Run optimization
    r = ro.optimization.optimize(gate, pulse, initial_params, num_steps=200, tol=1e-7, return_history=True)

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params)
    assert np.allclose(1 - fidelity, r.infidelity, rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=1e-7)
    assert r.infidelity == r.history[-1]


@pytest.mark.optimization
def test_adam_decay() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0.01)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

    # Initial parameters
    initial_params = (7.6, (-0.1,), (1.8, -0.6), ())

    # Run optimization
    r = ro.optimization.optimize(gate, pulse, initial_params, num_steps=200, tol=1e-7)

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params)
    assert np.allclose(1 - fidelity, r.infidelity, rtol=1e-12)


@pytest.mark.optimization
def test_multi_start_adam() -> None:
    tol = 1e-4

    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=2.0, decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const_cos_crab)

    # Parameter bounds for choosing random initial parameters
    min_initial_params = (6, (-2, -2, -2), (), ())
    max_initial_params = (9, (2, 2, 2), (), ())

    # Run optimization
    r = ro.optimization.multi_start_optimize(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=500,
        num_initializations=100,
        min_converged_initializations=2,
        tol=tol,
        return_all=True,
        num_processes=1,
        return_history=True,
    )

    # Verify the fidelities of the 'min_converged_initializations'
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params[0], tol=tol)
    assert np.allclose(1 - fidelity, r.infidelity[0], rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=tol)
    assert r.infidelity[0] == r.history[-1, 0]

    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params[1], tol=tol)
    assert np.allclose(1 - fidelity, r.infidelity[1], rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=tol)
    assert r.infidelity[1] == r.history[-1, 1]


@pytest.mark.optimization
def test_multi_start_adam_decay() -> None:
    tol = 1e-3

    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=2.0, decay=0.0005)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const_cos_crab)

    # Parameter bounds for choosing random initial parameters
    min_initial_params = (6, (-2, -2, -2), (), ())
    max_initial_params = (9, (2, 2, 2), (), ())

    # Run optimization
    r = ro.optimization.multi_start_optimize(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=100,
        num_initializations=20,
        tol=tol,
        num_processes=1,
    )

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params, tol=tol)
    assert np.allclose(1 - fidelity, r.infidelity, rtol=1e-12)


@pytest.mark.optimization
def test_fastest() -> None:
    tol = 1e-4

    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

    # Parameter bounds for choosing random initial parameters
    min_initial_params = (6, (-2,), (-2, -2), ())
    max_initial_params = (9, (2,), (2, 2), ())

    # Run optimization
    r = ro.optimization.multi_start_optimize(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=500,
        num_initializations=100,
        min_converged_initializations=20,
        num_processes=4,
        tol=tol,
        return_history=True,
    )

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params, tol=tol)
    assert np.allclose(1 - fidelity, r.infidelity, rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=tol)
    assert r.infidelity == r.history[-1]


@pytest.mark.optimization
def test_fixed() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

    # Initial parameters
    initial_params = (7.6, (0.0,), (1.8, -0.6), ())
    fixed_initial_params = (False, (True,), (False, False), ())

    # Run optimization
    r = ro.optimization.optimize(gate, pulse, initial_params, fixed_initial_params, num_steps=200)

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params)
    assert np.allclose(1 - fidelity, r.infidelity, rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=1e-7)
