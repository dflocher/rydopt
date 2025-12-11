from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from rydopt.pulses.general_pulse_ansatz_functions import const
from rydopt.types import PulseAnsatzFunction, PulseFunction, PulseParams


def _const_zero(t: jnp.ndarray | float, _duration: float, _ansatz_params: jnp.ndarray) -> jnp.ndarray:
    return const(t, _duration, jnp.array([0.0]))


def _const_one(t: jnp.ndarray | float, _duration: float, _ansatz_params: jnp.ndarray) -> jnp.ndarray:
    return const(t, _duration, jnp.array([1.0]))


@dataclass
class PulseAnsatz:
    r"""Data class that stores ansatz functions for the laser pulse that couples the qubit state :math:`|1\rangle` to
    the Rydberg state :math:`|r\rangle`.

    For available ansatz functions, see below. The parameters of the ansatz functions and duration of the laser pulse
    will be optimized to maximize the gate fidelity.

    Example:
        >>> import rydopt as ro
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )

    Attributes:
        detuning_ansatz: Detuning sweep, default is zero.
        phase_ansatz: Phase sweep, default is zero.
        rabi_ansatz: Rabi frequency amplitude sweep, default is one.

    """

    detuning_ansatz: PulseAnsatzFunction = _const_zero
    phase_ansatz: PulseAnsatzFunction = _const_zero
    rabi_ansatz: PulseAnsatzFunction = _const_one

    def make_pulse_functions(self, params: PulseParams) -> tuple[PulseFunction, PulseFunction, PulseFunction]:
        r"""Create three functions that describe the detuning sweep, the phase sweep, and the rabi sweep for fixed
        parameters.

        Args:
            params: pulse parameters

        Returns:
            Three functions :math:`\Delta(t), \, \xi(t), \, \Omega(t)`

        """
        duration, detuning_ansatz_params, phase_ansatz_params, rabi_ansatz_params = params
        detuning_ansatz_params = jnp.asarray(detuning_ansatz_params)
        phase_ansatz_params = jnp.asarray(phase_ansatz_params)
        rabi_ansatz_params = jnp.asarray(rabi_ansatz_params)

        def detuning_pulse(t):
            return self.detuning_ansatz(t, duration, detuning_ansatz_params)

        def phase_pulse(t):
            return self.phase_ansatz(t, duration, phase_ansatz_params)

        def rabi_pulse(t):
            return self.rabi_ansatz(t, duration, rabi_ansatz_params)

        return detuning_pulse, phase_pulse, rabi_pulse

    def evaluate_pulse_functions(
        self, t: jnp.ndarray | float, params: PulseParams
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""Evaluate the detuning sweep, the phase sweep, and the rabi sweep for fixed
        parameters at the given times.

        Args:
            t: time samples at which the functions are evaluated
            params: pulse parameters

        Returns:
            Three arrays of values for :math:`\Delta`, :math:`\xi`, :math:`\Omega`

        """
        duration, detuning_ansatz_params, phase_ansatz_params, rabi_ansatz_params = params
        detuning_ansatz_params = jnp.asarray(detuning_ansatz_params)
        phase_ansatz_params = jnp.asarray(phase_ansatz_params)
        rabi_ansatz_params = jnp.asarray(rabi_ansatz_params)

        return (
            self.detuning_ansatz(t, duration, detuning_ansatz_params),
            self.phase_ansatz(t, duration, phase_ansatz_params),
            self.rabi_ansatz(t, duration, rabi_ansatz_params),
        )
