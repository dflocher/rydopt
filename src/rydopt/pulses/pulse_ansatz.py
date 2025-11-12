from dataclasses import dataclass
from rydopt.pulses.pulse_ansatz_functions import PulseAnsatzFunction


@dataclass
class PulseAnsatz:
    r"""Object that stores ansatz functions for the laser pulse that couples the qubit state :math:`|1\rangle` to the Rydberg state :math:`|r\rangle`.

    For available ansatz functions, see below. The parameters of the ansatz functions and duration of the laser pulse will be optimized to maximize the gate fidelity.
    """

    detuning_ansatz: PulseAnsatzFunction | None = None
    """Detuning sweep, or ``None`` if the detuning is zero."""

    phase_ansatz: PulseAnsatzFunction | None = None
    """Phase sweep, or ``None`` if the phase is zero."""

    rabi_ansatz: PulseAnsatzFunction | None = None
    """Rabi amplitude sweep, or ``None`` if the amplitude is one."""
