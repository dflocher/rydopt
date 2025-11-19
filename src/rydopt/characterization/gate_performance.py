from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation import evolve, evolve_TR
from typing import TypeAlias

FloatParams: TypeAlias = float | tuple[float, ...]


def analyze_gate(
    gate: Gate,
    pulse_ansatz: PulseAnsatz,
    params: tuple[FloatParams, ...],
):
    final_states = evolve(gate, pulse_ansatz, params)
    infid = 1 - gate.process_fidelity(final_states)

    decay = gate.get_decay()
    gate.set_decay(0.0)
    final_states_nodecay = evolve(gate, pulse_ansatz, params)
    infid_nodecay = 1 - gate.process_fidelity(final_states_nodecay)

    ryd_times_subsystems = evolve_TR(gate, pulse_ansatz, params)
    ryd_time = gate.rydberg_time(ryd_times_subsystems)
    gate.set_decay(decay)

    return infid, infid_nodecay, ryd_time
