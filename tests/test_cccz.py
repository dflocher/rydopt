import numpy as np
import rydopt as ro


def test_cccz() -> None:
    # number of atoms participating in the gate (2, 3 or 4)
    n_atoms = 4

    # Rydberg interaction strengths (use float("inf") for perfect blockade)
    Vnn = float("inf")
    Vnnn = float("inf")
    decay = 0.000

    # target gate phases
    theta = np.pi  # set theta=None, eps=None if both are arbitrary / set theta='eps', eps=None if both are arbitrary but must be equal
    eps = np.pi  # set eps=None if eps is arbitrary
    lamb = np.pi
    delta = np.pi  # set delta=None if delta is arbitrary
    kappa = np.pi

    # pulse type
    pulse = ro.pulses.pulse_phase_sin_crab

    # initial parameters
    initial_params = np.array([12.4, 0.1, 1.0, -1.0, 2.0, -0.8, 0.7, -0.2, 0.7, 0.3])

    # optimization settings
    N_epochs = 500
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
            12.42436209,
            0.09580844,
            1.01592733,
            -1.00783188,
            2.07969005,
            -0.80292432,
            0.71405035,
            -0.16563671,
            0.72792360,
            0.32205233,
        ]
    )
    assert np.allclose(params, ref, rtol=1e-4)
