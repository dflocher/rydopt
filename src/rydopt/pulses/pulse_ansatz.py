from dataclasses import dataclass
import jax.numpy as jnp
from rydopt.pulses.pulse_ansatz_functions import PulseAnsatzFunction, const


def const_zero(
    t: jnp.ndarray | float, _duration: float, _params: jnp.ndarray
) -> jnp.ndarray:
    return const(t, _duration, jnp.array([0.0]))


def const_one(
    t: jnp.ndarray | float, _duration: float, _params: jnp.ndarray
) -> jnp.ndarray:
    return const(t, _duration, jnp.array([1.0]))


@dataclass
class PulseAnsatz:
    r"""Data class that stores ansatz functions for the laser pulse that couples the qubit state :math:`|1\rangle` to the Rydberg state :math:`|r\rangle`.

    For available ansatz functions, see below. The parameters of the ansatz functions and duration of the laser pulse will be optimized to maximize the gate fidelity.

    Example:
        >>> import rydopt as ro
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )

    Attributes:
        detuning_ansatz (PulseAnsatzFunction): Detuning sweep, default is zero.
        phase_ansatz (PulseAnsatzFunction): Phase sweep, default is zero.
        rabi_ansatz (PulseAnsatzFunction): Rabi amplitude sweep, default is one.

    """

    detuning_ansatz: PulseAnsatzFunction = const_zero

    phase_ansatz: PulseAnsatzFunction = const_zero

    rabi_ansatz: PulseAnsatzFunction = const_one
