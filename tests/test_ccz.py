import numpy as np
import rydopt as ro
import pytest


@pytest.mark.optimization
def test_ccz() -> None:
    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 3

    # Rydberg interaction strengths (use float("inf") for perfect blockade)
    Vnn = float("inf")
    Vnnn = float("inf")
    decay = 0.000

    # target gate phases
    theta = np.pi  # set theta=None, eps=None if both are arbitrary / set theta='eps', eps=None if both are arbitrary but must be equal
    eps = np.pi  # set eps=None if eps is arbitrary
    lamb = np.pi
    delta = 0  # set delta=None if delta is arbitrary
    kappa = 0

    # pulse type
    pulse = ro.pulses.pulse_phase_sin_crab

    # initial parameters
    initial_params = np.array([11.0, 0.2, 0.4, -1.2, 1.1, -0.7, 0.9, -0.2])

    # optimization settings
    N_epochs = 200
    learning_rate = 0.05
    T_penalty = 0.0

    # run optimization
    params = ro.optimization.train_single_gate(
        n_atoms,
        Vnn,
        Vnnn,
        theta,
        eps,
        lamb,
        delta,
        kappa,
        pulse,
        initial_params,
        N_epochs,
        learning_rate,
        T_penalty,
        decay,
    )

    # compare result to reference
    ref = np.array(
        [
            10.99552491,
            0.20352068,
            0.43322811,
            -1.18878954,
            1.10057937,
            -0.70670388,
            1.16454156,
            -0.25082207,
        ]
    )
    assert np.allclose(params, ref, rtol=1e-4)
