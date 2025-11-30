import numpy as np
import rydopt as ro

if __name__ == "__main__":
    # A CZ gate on two atoms in the perfect Rydberg blockade regime; the atoms exhibit
    # Rydberg state decay
    gate = ro.gates.TwoQubitGate(
        phi=None,
        theta=np.pi,
        Vnn=float("inf"),
        decay=0.0001,
    )

    # Pulse ansatz: constant detuning, sweep of laser phase according to sin_crab ansatz
    pulse_ansatz = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab
    )

    # Pulse parameters
    pulse_params = (7.61140652, (-0.07842706,), (1.80300902, -0.61792703), ())

    # Plot the pulse
    ro.characterization.plot_pulse(pulse_ansatz, pulse_params)

    # Determine the gate's infidelity, infidelity without decay, and Rydberg time
    infidelity, infidelity_nodecay, ryd_time = ro.characterization.analyze_gate(
        gate, pulse_ansatz, pulse_params
    )

    # Determine the gate's infidelity, infidelity without decay, and Rydberg time using
    # the full Hamiltonian and qutip
    infidelity_qutip, infidelity_nodecay_qutip, ryd_time_qutip = (
        ro.characterization.analyze_gate_qutip(gate, pulse_ansatz, pulse_params)
    )

    # Print the gate performance measures
    print(
        f"Gate infidelity:             jax: {infidelity:.4e}, "
        f"qutip: {infidelity_qutip:.4e}"
    )
    print(
        f"Gate infidelity (no decay):  jax: {infidelity_nodecay:.4e}, "
        f"qutip: {infidelity_nodecay_qutip:.4e}"
    )
    print(
        f"Rydberg time:                jax: {ryd_time:.4f},     "
        f"qutip: {ryd_time_qutip:.4f}"
    )
