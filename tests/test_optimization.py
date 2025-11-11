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
    T_penalty = 0.0
    N_epochs = 200
    learning_rate = 0.05

    # run optimization
    params = ro.optimization.adam(
        gate,
        pulse,
        initial_params,
        T_penalty,
        N_epochs,
        learning_rate,
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
    assert np.allclose(params, ref, rtol=1e-1)


@pytest.mark.optimization
def test_multi_start_adam() -> None:
    # gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # pulse type
    pulse = ro.pulses.pulse_phase_sin_crab

    # optimization settings
    T_default = 11.0
    T_penalty = 0.0
    N_searches = 2
    N_params = 4
    N_epochs = 200
    learning_rate = 0.05

    # run optimization
    _ = ro.optimization.multi_start_adam(
        gate,
        pulse,
        T_default,
        T_penalty,
        N_searches,
        N_params,
        N_epochs,
        learning_rate,
    )
