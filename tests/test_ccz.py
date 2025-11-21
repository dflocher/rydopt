import numpy as np
import rydopt as ro
import pytest


@pytest.mark.optimization
def test_ccz() -> None:
    # Gate
    gate = ro.gates.ThreeQubitGateIsosceles(
        phi=None,
        theta=np.pi,
        eps=np.pi,
        lamb=np.pi,
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
    initial_params = (11.0, (0.2,), (0.4, -1.2, 1.1, -0.7, 0.9, -0.2), ())

    # Run optimization
    r = ro.optimization.adam(gate, pulse, initial_params, num_steps=200, tol=1e-7)

    # Compare result to reference
    ref = (
        10.99552491,
        (0.20352,),
        (0.43322811, -1.18878954, 1.10057937, -0.70670388, 1.16454156, -0.25082207),
        (),
    )
    assert all(np.allclose(x, y, rtol=1e-3) for x, y in zip(r.params, ref))
