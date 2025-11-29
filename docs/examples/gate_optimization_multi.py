import numpy as np

import rydopt as ro

if __name__ == "__main__":
    # Want to perform a CZ gate on two atoms with finite interaction; no Rydberg state decay
    gate = ro.gates.TwoQubitGate(
        phi=None,
        theta=np.pi,
        Vnn=float("inf"),
        decay=0.0,
    )

    # Pulse ansatz: constant detuning, sweep of the laser phase according to sin_crab ansatz
    pulse_ansatz = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

    # Bounds for the initial pulse parameter guesses
    min_initial_params = (6, (-2,), (-2, -2), ())
    max_initial_params = (9, (2,), (2, 2), ())

    # Optimize the pulse parameters
    opt_result = ro.optimization.multi_start_optimize(
        gate,
        pulse_ansatz,
        min_initial_params,
        max_initial_params,
        tol=1e-7,
        num_initializations=100,
        num_processes=None,
        return_history=True,
    )
    optimized_params = opt_result.params

    # Determine the gate's infidelity, infidelity without decay, and Rydberg time using
    # the subsystem Hamiltonians and jax
    infidelity, infidelity_nodecay, ryd_time = ro.characterization.analyze_gate(gate, pulse_ansatz, optimized_params)

    # Print the gate performance measures
    print("\n=== Performance analysis of the best/fastest optimized gate pulse ===\n")
    print(f"Gate infidelity:             {infidelity:.4e}")
    print(f"Gate infidelity (no decay):  {infidelity_nodecay:.4e}")
    print(f"Rydberg time:                {ryd_time:.4f}")

    ro.characterization.plot_optimization_history(opt_result)
