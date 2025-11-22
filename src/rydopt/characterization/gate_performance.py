from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.simulation.rydberg_time import rydberg_time
from rydopt.types import FloatParams


def analyze_gate(
    gate: Gate,
    pulse_ansatz: PulseAnsatz,
    params: tuple[FloatParams, ...],
    tol: float = 1e-15,
):
    gate_nodecay = gate.copy()
    gate_nodecay.set_decay(0.0)

    infidelity = 1 - process_fidelity(gate, pulse_ansatz, params, tol=tol)
    infidelity_nodecay = 1 - process_fidelity(
        gate_nodecay, pulse_ansatz, params, tol=tol
    )
    ryd_time = rydberg_time(gate_nodecay, pulse_ansatz, params, tol=tol)

    return infidelity, infidelity_nodecay, ryd_time
