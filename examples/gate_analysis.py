import rydopt as ro
import numpy as np

if __name__ == "__main__":
    # Gate
    gate = ro.gates.TwoQubitGate(
        phi=None,
        theta=np.pi,
        Vnn=float("inf"),
        decay=0.0001,
    )

    # Pulse ansatz
    pulse_ansatz = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab
    )

    # Pulse parameters
    pulse_params = (7.6, (-0.1,), (1.8, -0.6), ())

    # Plot the pulse
    ro.characterization.plot_pulse(pulse_ansatz, pulse_params)

    # Determine the gate's infidelity, infidelity without decay, and Rydberg time using the subsystem Hamiltonians and jax
    infidelity, infidelity_nodecay, ryd_time = ro.characterization.analyze_gate(
        gate, pulse_ansatz, pulse_params
    )

    # Determine the gate's infidelity, infidelity without decay, and Rydberg time using the full Hamiltonian and qutip
    infidelity_qutip, infidelity_nodecay_qutip, ryd_time_qutip = (
        ro.characterization.analyze_gate_qutip(gate, pulse_ansatz, pulse_params)
    )

    # Print the gate performance measures
    print(
        f"Gate infidelity:             jax: {infidelity:.4e}, qutip: {infidelity_qutip:.4e}"
    )
    print(
        f"Gate infidelity (no decay):  jax: {infidelity_nodecay:.4e}, qutip: {infidelity_nodecay_qutip:.4e}"
    )
    print(
        f"Rydberg time:                jax: {ryd_time:.4f},    qutip: {ryd_time_qutip:.4f}"
    )
