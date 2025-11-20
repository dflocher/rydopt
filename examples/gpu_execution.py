"""The example takes
* ~70m on a AMD Ryzen 7 5700G CPU
* 9m 17.916s on a NVIDIA GeForce RTX 4060 Ti GPU
* 4m 54.621s on a NVIDIA H100 PCIe GPU
"""

import jax

jax.config.update("jax_platforms", "cuda,cpu")
import rydopt as ro  # noqa: E402
import numpy as np  # noqa: E402

# Gate
gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=1.5, decay=0)

# Pulse
pulse = ro.pulses.PulseAnsatz(
    detuning_ansatz=ro.pulses.const,
    phase_ansatz=ro.pulses.sin_crab,
    rabi_ansatz=None,
)

# Parameter bounds for choosing random initial parameters
min_initial_params = (6, (-2,), (-2, -2, -2, -2, -2, -2), ())
max_initial_params = (9, (2,), (2, 2, 2, 2, 2, 2), ())

# Run optimization
params = ro.optimization.multi_start_adam(
    gate,
    pulse,
    min_initial_params,
    max_initial_params,
    num_steps=300,
    num_initializations=10000,
    min_converged_initializations=5000,
    tol=1e-7,
)

# Print the parameters that belong to the fastest gate
print(f"Parameters of the fastest gate: {params}")
