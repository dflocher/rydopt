import jax

jax.config.update("jax_platforms", "cuda,cpu")
import rydopt as ro  # noqa: E402
import numpy as np  # noqa: E402


# Gate
# gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=1.5, decay=0.0001)
# gate = ro.gates.ThreeQubitGateIsosceles(phi=None, theta=np.pi, eps=None, lamb=np.pi, Vnn=float("inf"), Vnnn=0.5, decay=0.0001)
gate = ro.gates.FourQubitGatePyramidal(
    phi=None,
    theta=np.pi,
    eps=None,
    lamb=np.pi,
    delta=0.0,
    kappa=0.0,
    Vnn=float("inf"),
    Vnnn=0.5,
    decay=0.0001,
)

# Pulse ansatz
pulse_ansatz = ro.pulses.PulseAnsatz(
    detuning_ansatz=ro.pulses.const,
    phase_ansatz=ro.pulses.sin_crab,
    rabi_ansatz=None,
)

# Pulse parameters
pulse_params = (7.6, (-0.1,), (1.8, -0.6), ())

# Determine the gate's infidelity, infidelity without decay, and Rydberg time using the subsystem Hamiltonians and jax
infid, infid_nodecay, ryd_time = ro.characterization.analyze_gate(
    gate, pulse_ansatz, pulse_params
)

# Determine the gate's infidelity, infidelity without decay, and Rydberg time using the full Hamiltonian and qutip
infid_qutip, infid_nodecay_qutip, ryd_time_qutip = (
    ro.characterization.analyze_gate_qutip(gate, pulse_ansatz, pulse_params)
)

# Print the gate performance measures
print(
    "Gate infidelity:             jax: {i:.4e}, qutip: {i_q:.4e}".format(
        i=infid, i_q=infid_qutip
    )
)
print(
    "Gate infidelity (no decay):  jax: {i_n:.4e}, qutip: {i_n_q:.4e}".format(
        i_n=infid_nodecay, i_n_q=infid_nodecay_qutip
    )
)
print(
    "Rydberg time:                jax: {r:.5f},    qutip: {r_q:.5f}".format(
        r=ryd_time, r_q=ryd_time_qutip
    )
)
