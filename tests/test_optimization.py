import numpy as np
import rydopt as ro
import pytest


@pytest.mark.optimization
def test_adam() -> None:
    # gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # pulse type
    pulse = ro.pulses.pulse_phase_sin_crab

    # initial parameters
    initial_params = np.array([7.6, -0.1, 1.8, -0.6])

    # optimization settings
    N_epochs = 200
    learning_rate = 0.05
    T_penalty = 0.0

    # run optimization
    params = ro.optimization.adam(
        gate,
        pulse,
        initial_params,
        N_epochs,
        learning_rate,
        T_penalty,
    )

    # compare result to reference
    ref = np.array(
        [
            7.61141034,
            -0.07884777,
            1.83253308,
            -0.61765787,
        ]
    )
    assert np.allclose(params, ref, rtol=1e-4)


@pytest.mark.optimization
def test_multi_start_adam() -> None:
    # gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # pulse type
    pulse = ro.pulses.pulse_phase_sin_crab

    # parameter bounds for choosing random initial parameters
    min_initial_params = np.array([6, -1, -2, -2])
    max_initial_params = np.array([9, 1, 2, 2])

    # optimization settings
    N_searches = 5
    N_epochs = 200
    learning_rate = 0.05
    T_penalty = 0.0

    # run optimization
    _ = ro.optimization.multi_start_adam(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        N_searches,
        N_epochs,
        learning_rate,
        T_penalty,
    )
