import numpy as np
import rydopt as ro
import pytest


@pytest.mark.optimization
def test_cz() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab
    )

    # Initial parameters
    initial_params = (7.6, (-0.1,), (1.8, -0.6), ())

    # Run optimization
    r = ro.optimization.adam(gate, pulse, initial_params, num_steps=200, tol=1e-7)

    # Compare result to reference
    ref = (7.61141034, (-0.07884777,), (1.83253308, -0.61765787), ())
    assert all(np.allclose(x, y, rtol=1e-3) for x, y in zip(r.params, ref))  # type: ignore[arg-type]
