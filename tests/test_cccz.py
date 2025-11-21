import numpy as np
import rydopt as ro
import pytest


@pytest.mark.optimization
def test_cccz() -> None:
    # Gate
    gate = ro.gates.FourQubitGatePyramidal(
        phi=None,
        theta=np.pi,
        eps=np.pi,
        lamb=np.pi,
        delta=np.pi,
        kappa=np.pi,
        Vnn=float("inf"),
        Vnnn=float("inf"),
        decay=0.0,
    )

    # Pulse
    pulse = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.sin_crab,
    )

    # Initial parameters
    initial_params = (
        12.4,
        (0.1,),
        (1.0, -1.0, 2.0, -0.8, 0.7, -0.2, 0.7, 0.3),
        (),
    )

    # Run optimization
    r = ro.optimization.adam(gate, pulse, initial_params, num_steps=500, tol=1e-7)

    # Compare result to reference
    ref = (
        12.42436209,
        (0.09580844,),
        (
            1.01592733,
            -1.00783188,
            2.07969005,
            -0.80292432,
            0.71405035,
            -0.16563671,
            0.72792360,
            0.32205233,
        ),
        (),
    )
    assert all(np.allclose(x, y, rtol=1e-2) for x, y in zip(r.params, ref))
