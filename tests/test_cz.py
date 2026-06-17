import numpy as np
import pytest

import rydopt as ro
from rydopt.pulses.pulse_ansatz import pack_params


@pytest.mark.optimization
def test_cz() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.Const(), phase_ansatz=ro.pulses.SinCrab(2))

    # Initial parameters
    initial_params = ro.pulses.PulseParams(7.6, [0.1], [1.8, -0.6], [])

    # Run optimization
    r = ro.optimization.optimize(gate, pulse, initial_params, num_steps=200, tol=1e-7)

    # Compare result to reference
    ref = np.array([7.61141034, 0.07884777, 1.83253308, -0.61765787])
    estimated = pack_params(r.params)

    assert np.allclose(estimated, ref, rtol=1e-3)
