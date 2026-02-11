from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import jax
import jax.numpy as jnp

from rydopt.protocols import Evolvable
from rydopt.types import GenericPulseParams, PulseAnsatzFunction, PulseFunction

from .pulse_ansatz import PulseAnsatz, _const_one, _const_zero


class PulseParamMap(Protocol):
    def map_duration(self, phase: float | jnp.ndarray, pulse_params: GenericPulseParams) -> float | jnp.ndarray: ...

    def map_full(
        self, phase: float | jnp.ndarray, pulse_params: GenericPulseParams
    ) -> tuple[float | jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: ...


@dataclass
class PolynomialPulseMap(PulseParamMap):
    r"""Data class that stores a polynomial map of ansatz parameters. The map converts the
    trainable pulse parameters into ansatz parameters given the target phase of the gate.

    Args:
        degrees: the polynomial's degrees
        num_params: the number of parameters for duration, detuning, phase, and Rabi frequency

    Returns:
        pulse parameters

    """

    degrees: Sequence[int] = field(default_factory=lambda: [0, 0, 0, 0])
    num_params: Sequence[int] = field(default_factory=lambda: [1, 1, 1, 1])

    def _poly_eval(self, phase: jnp.ndarray, params: jnp.ndarray, degree: int):
        """Efficient polynomial evaluation."""
        if degree == 0:
            return params[0]

        powers = jnp.power(phase, jnp.arange(degree + 1))
        return jnp.dot(params, powers)

    def map_duration(
        self,
        phase: float | jnp.ndarray,
        pulse_params: GenericPulseParams,
    ) -> float | jnp.ndarray:
        phase = jnp.asarray(phase)
        params = jnp.ravel(jnp.asarray(pulse_params[0]))
        degree = self.degrees[0]

        duration = self._poly_eval(phase, params, degree)
        return jax.nn.softplus(duration) if degree > 0 else duration

    def map_full(
        self,
        phase: float | jnp.ndarray,
        pulse_params: GenericPulseParams,
    ) -> tuple[float | jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        phase = jnp.asarray(phase)

        # --- Duration ---
        duration = self.map_duration(phase, pulse_params)

        # --- Remaining parameters ---
        outputs = []

        for i in range(1, 4):
            degree = self.degrees[i]
            params = jnp.asarray(pulse_params[i])

            if degree == 0:
                outputs.append(jnp.ravel(params))
                continue

            coeffs = params.reshape(self.num_params[i], degree + 1)
            powers = jnp.power(phase, jnp.arange(degree + 1))
            outputs.append(coeffs @ powers)

        detuning, laser_phase, rabi = outputs

        return duration, detuning, laser_phase, rabi


@dataclass
class MappedPulseAnsatz(PulseAnsatz):
    r"""Data class that stores ansatz functions for the laser pulse.
     The parameters of the ansatz functions and duration of the laser pulse will be optimized to
     maximize the gate fidelity. Here, the parameters of one of the ansatze are treated as
     fix-degree polynomials in the parametrized phase of the gate.

    Example:
        >>> import rydopt as ro
        >>> degrees = [0, 0, 3, 0]
        >>> num_params = [1, 1, 10, 1]
        >>> pulse_map = ro.pulses.PolynomialPulseMap(degrees, num_params)
        >>> pulse = ro.pulses.MappedPulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ...     pulse_map=pulse_map,
        ... )

    Attributes:
        detuning_ansatz: Detuning sweep, default is zero.
        phase_ansatz: Phase sweep, default is zero.
        rabi_ansatz: Rabi frequency amplitude sweep, default is one.
        pulse_map: a map from an array of pulse parameters to the selcted ansatz parameters
            given a target phase of the gate, default is PolynomialPulseMap.

    """

    detuning_ansatz: PulseAnsatzFunction = _const_zero
    phase_ansatz: PulseAnsatzFunction = _const_zero
    rabi_ansatz: PulseAnsatzFunction = _const_one
    pulse_map: PulseParamMap = field(default_factory=lambda: PolynomialPulseMap())

    @staticmethod
    def target_phase(gate: Evolvable) -> float | jnp.ndarray:
        theta = getattr(gate, "_theta", None)
        if theta is None:
            raise ValueError("Gate must define non-None '_theta'.")
        return cast(jnp.ndarray, theta) / (2 * jnp.pi)

    def generate_pulse_params(
        self, gate: Evolvable, params: GenericPulseParams
    ) -> tuple[float | jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        phase = self.target_phase(gate)
        return self.pulse_map.map_full(phase, params)

    def generate_duration(self, gate: Evolvable, params: GenericPulseParams) -> float | jnp.ndarray:
        phase = self.target_phase(gate)
        return self.pulse_map.map_duration(phase, params)

    def make_pulse_functions_for_gate(
        self, params: GenericPulseParams, gate: Evolvable
    ) -> tuple[PulseFunction, PulseFunction, PulseFunction]:
        r"""Create three functions that describe the detuning sweep, the phase sweep, and the rabi sweep for fixed
        parameters for a parametrized gate with a predefined target phase.

        Args:
            params: pulse parameters
            gate: a gate instance

        Returns:
            Three functions :math:`\Delta(t), \, \xi(t), \, \Omega(t)`

        """
        (
            duration,
            detuning_ansatz_params,
            phase_ansatz_params,
            rabi_ansatz_params,
        ) = self.generate_pulse_params(gate, params)

        def detuning_pulse(t):
            return self.detuning_ansatz(t, duration, detuning_ansatz_params)

        def phase_pulse(t):
            return self.phase_ansatz(t, duration, phase_ansatz_params)

        def rabi_pulse(t):
            return self.rabi_ansatz(t, duration, rabi_ansatz_params)

        return detuning_pulse, phase_pulse, rabi_pulse

    def evaluate_pulse_functions_for_gate(
        self, t: jnp.ndarray | float, params: GenericPulseParams, gate: Evolvable
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""Evaluate the detuning sweep, the phase sweep, and the rabi sweep for fixed
        parameters at the given times for a parametrized gate with a predefined target phase.

        Args:
            t: time samples at which the functions are evaluated
            params: pulse parameters
            gate: gate instance

        Returns:
            Three arrays of values for :math:`\Delta`, :math:`\xi`, :math:`\Omega`

        """
        (
            duration,
            detuning_ansatz_params,
            phase_ansatz_params,
            rabi_ansatz_params,
        ) = self.generate_pulse_params(gate, params)
        return (
            self.detuning_ansatz(t, duration, detuning_ansatz_params),
            self.phase_ansatz(t, duration, phase_ansatz_params),
            self.rabi_ansatz(t, duration, rabi_ansatz_params),
        )
