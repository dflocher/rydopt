import numpy as np
import rydopt as ro
import pytest


@pytest.mark.optimization
def test_cz() -> None:
    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 2

    # Rydberg interaction strengths (use float("inf") for perfect blockade)
    Vnn = float("inf")
    Vnnn = float("inf")
    decay = 0.000

    # target gate phases
    theta = np.pi  # set theta=None, eps=None if both are arbitrary / set theta='eps', eps=None if both are arbitrary but must be equal
    eps = 0  # set eps=None if eps is arbitrary
    lamb = 0
    delta = 0  # set delta=None if delta is arbitrary
    kappa = 0

    # pulse type
    pulse = ro.pulses.pulse_phase_sin_crab

    # initial parameters
    initial_params = np.array([7.6, -0.1, 1.8, -0.6])

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

    # verify the fidelity
    fidelity, _ = ro.characterization.verify(
        n_atoms,
        Vnn,
        Vnnn,
        theta,
        eps,
        lamb,
        delta,
        kappa,
        "pulse_phase_sin_crab",
        params,
        decay,
    )
    assert np.allclose(fidelity, 1, rtol=1e-6)

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
