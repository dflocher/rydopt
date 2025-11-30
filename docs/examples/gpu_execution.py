"""The example takes

* 29m 28,732s on AMD Ryzen 7 5700G
* 4m 49.323s on AMD EPYC 9374F
* 4m 20.041s on NVIDIA GeForce RTX 4060 Ti
* 2m 10.97s on NVIDIA H100 PCIe
"""

import os

os.environ["JAX_PLATFORMS"] = "cuda,cpu"
import numpy as np
import rydopt as ro

if __name__ == "__main__":
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=1.5, decay=0)

    # Pulse
    pulse = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab
    )

    # Parameter bounds for choosing random initial parameters
    min_initial_params = (6, (-2,), (-2, -2, -2, -2, -2, -2), ())
    max_initial_params = (9, (2,), (2, 2, 2, 2, 2, 2), ())

    # Run optimization
    _ = ro.optimization.multi_start_optimize(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=300,
        num_initializations=10000,
        min_converged_initializations=5000,
        tol=1e-7,
    )
