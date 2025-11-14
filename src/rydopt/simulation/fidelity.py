from rydopt.gates.gate import Gate
from rydopt.simulation.evolve import evolve
from rydopt.pulses.pulse_ansatz import PulseAnsatz


def process_fidelity(gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7):
    final_states = evolve(gate, pulse, params, tol)
    return gate.process_fidelity(final_states)


def average_gate_fidelity(
    gate: Gate, pulse: PulseAnsatz, params: tuple, tol: float = 1e-7
):
    return (gate.dim() * process_fidelity(gate, pulse, params, tol) + 1.0) / (
        gate.dim() + 1.0
    )
