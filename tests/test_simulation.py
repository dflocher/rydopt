import time

import jax
import numpy as np

import rydopt as ro


def test_fidelity_calculation() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(
        detuning_ansatz=ro.pulses.Const(),
        phase_ansatz=ro.pulses.SinCrab(2),
    )

    # Parameters
    params = ro.pulses.PulseParams(7.61141034, [0.078847771], [1.83253308, -0.61765787], [])

    # Time evolve the system and get the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, params)
    assert np.allclose(fidelity, 1, rtol=1e-6)


def test_evolution_performance() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(
        detuning_ansatz=ro.pulses.Const(),
        phase_ansatz=ro.pulses.SinCrab(2),
    )

    # Parameters
    params = ro.pulses.PulseParams(7.61141034, [0.078847771], [1.83253308, -0.61765787], [])

    # Compilation and warmup
    evolve_jit = jax.jit(lambda params: ro.simulation.evolve(gate, pulse, params))
    result = evolve_jit(params)
    _ = jax.block_until_ready(result)

    # Measure the performance
    num_runs = 1000

    t0 = time.perf_counter()
    for i in range(num_runs):
        result = evolve_jit(ro.pulses.PulseParams(params[0] + i * 0.001, params[1], params[2], params[3]))
        _ = jax.block_until_ready(result)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1e3 / num_runs
    print(f"Average duration of evolve_jit: {avg_ms:.4f} ms over {num_runs} runs")
