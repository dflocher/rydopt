from dataclasses import dataclass
import jax.numpy as jnp
from rydopt.pulses.pulse_ansatz_functions import PulseAnsatzFunction, const
from rydopt.types import FloatParams


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
        detuning_ansatz: Detuning sweep, default is zero.
        phase_ansatz: Phase sweep, default is zero.
        rabi_ansatz: Rabi amplitude sweep, default is one.

    """

    detuning_ansatz: PulseAnsatzFunction = const_zero

    phase_ansatz: PulseAnsatzFunction = const_zero

    rabi_ansatz: PulseAnsatzFunction = const_one

    def make_pulses(self, params: tuple[FloatParams, ...]) -> tuple:
        r"""
        Create three functions that describe the detuning sweep, the phase sweep, and the rabi sweep.

        Args:
            params: pulse parameters

        Returns:
            Three functions :math:`\Delta(t), \, \xi(t), \, |\Omega(t)|`
        """
        T, detuning_params, phase_params, rabi_params = params
        detuning_params = jnp.asarray(detuning_params)
        phase_params = jnp.asarray(phase_params)
        rabi_params = jnp.asarray(rabi_params)
        detuning_pulse = lambda t: self.detuning_ansatz(t, T, detuning_params)  # noqa: E731
        phase_pulse = lambda t: self.phase_ansatz(t, T, phase_params)  # noqa: E731
        rabi_pulse = lambda t: self.rabi_ansatz(t, T, rabi_params)  # noqa: E731
        return detuning_pulse, phase_pulse, rabi_pulse
