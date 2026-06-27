from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from rydopt.pulses.pulse_ansatz import PulseAnsatz, PulseAnsatzFunction, _FixedConstant, _is_unpacked
from rydopt.pulses.pulse_family_params import PulseFamilyParams
from rydopt.pulses.pulse_params import PulseParams
from rydopt.types import ParamsFloatLike


@runtime_checkable
class PulseParamMap(Protocol):
    """Minimal interface for the map of ansatz parameters used in PulseFamilyAnsatz."""

    def map_duration(
        self,
        target_phase: float | jax.Array,
        packed_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> float | jax.Array: ...

    def map_full(
        self,
        target_phase: float | jax.Array,
        packed_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...

    def map_shape(
        self,
        params_count: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...


@dataclass
class PolynomialPulseMap:
    r"""Polynomial map of ansatz parameters.

    Converts trainable pulse parameters into ansatz parameters given the
    target phase of the gate. Each component is treated as a polynomial in the
    target phase, with per-component degree given by ``degrees``.

    Args:
        degrees: polynomial degree for
            ``(duration, detuning, phase, rabi)``.

    """

    degrees: Sequence[int] = field(default_factory=lambda: [0, 0, 0, 0])

    @staticmethod
    def _poly_eval(target_phase: jax.Array, coeffs: jax.Array, degree: int) -> jax.Array:
        """Evaluate a polynomial in ``target_phase``.

        Works for 1-D ``coeffs`` of shape ``(degree + 1,)`` returning a scalar,
        and for 2-D ``coeffs`` of shape ``(n, degree + 1)`` returning shape ``(n,)``.
        """
        powers = jnp.power(target_phase, jnp.arange(degree + 1))
        return coeffs @ powers

    def map_duration(
        self,
        target_phase: float | jax.Array,
        packed_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        target_phase = jnp.asarray(target_phase)
        params = jnp.ravel(jnp.asarray(packed_params[0]))
        degree = self.degrees[0]

        if degree > 0:
            duration = self._poly_eval(target_phase, params, degree)
            return jax.nn.softplus(duration)
        return params[0]

    def map_full(
        self,
        target_phase: float | jax.Array,
        packed_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        target_phase = jnp.asarray(target_phase)

        # --- Duration ---
        duration = self.map_duration(target_phase, packed_params)

        # --- Remaining parameters (detuning, phase, rabi) ---
        outputs = []
        for i in range(1, 4):
            degree = self.degrees[i]
            params = jnp.asarray(packed_params[i])
            if degree == 0:
                outputs.append(jnp.ravel(params))
                continue

            if params.size % (degree + 1) != 0:
                raise ValueError(f"Cannot reshape {params.size} elements into rows of size {degree + 1}")

            coeffs = params.reshape(-1, degree + 1)
            outputs.append(self._poly_eval(target_phase, coeffs, degree))

        detuning, laser_phase, rabi = outputs

        return (
            duration,
            jnp.asarray(detuning),
            jnp.asarray(laser_phase),
            jnp.asarray(rabi),
        )

    @staticmethod
    def shape(count: int, degree: int) -> tuple[int, ...]:
        if degree == 0:
            return (count,)
        return count, degree + 1

    def map_shape(
        self,
        params_count: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (
            (self.degrees[0] + 1,),
            self.shape(params_count[0], self.degrees[1]),
            self.shape(params_count[1], self.degrees[2]),
            self.shape(params_count[2], self.degrees[3]),
        )


@dataclass
class PulseFamilyAnsatz:
    r"""Data class that stores ansatz functions for a family of laser pulses.

    A pulse family describes a continuous family of gates parameterized by a gate parameter :math:`\phi`
    (for example, the target phase of a controlled phase gate). Rather than optimizing an independent pulse
    for each value of :math:`\phi`, the pulse duration and ansatz parameters are represented as functions of
    :math:`\phi`.

    RydOpt models this dependence through a parameter map. The packed pulse parameters are optimized once
    and mapped to the pulse duration and ansatz parameters for a specific gate parameter value.
    By default, :class:`PolynomialPulseMap` represents each pulse parameter as a polynomial of fixed degree
    in :math:`\phi`.

    For available ansatz functions for the detuning :math:`\Delta(t)`, phase :math:`\xi(t)`,
    and Rabi frequency :math:`\Omega(t)` sweeps, see below. The function :func:`optimize_family
    <rydopt.optimization.optimize>` allows optimizing the pulse-family parameters to maximize
    fidelity across a target gate family. Initial pulse-family parameters can be provided as
    ``PulseFamilyParams(duration_params, detuning_params, phase_params, rabi_params)``,
    where each array contains the coefficients used by ``pulse_map`` to construct the corresponding
    pulse duration or ansatz parameters for a given gate parameter value.

    Example:
        >>> import rydopt as ro
        >>> degrees = [2, 0, 3, 0]
        >>> num_phase_params = 10
        >>> pulse_map = ro.pulses.PolynomialPulseMap(degrees)
        >>> pulse_family = ro.pulses.PulseFamilyAnsatz(
        ...     detuning_ansatz=ro.pulses.Const(),
        ...     phase_ansatz=ro.pulses.SinCrab(num_phase_params),
        ...     pulse_map=pulse_map,
        ... )

    Attributes:
        detuning_ansatz:
            Detuning sweep :math:`\Delta(t)`, default is zero.
        phase_ansatz:
            Phase sweep :math:`\xi(t)`, default is zero.
        rabi_ansatz:
            Rabi frequency amplitude sweep :math:`\Omega(t)`, default is one.
        pulse_map:
            Maps optimized pulse-family parameters to the pulse duration and
            ansatz parameters for a given gate parameter value. The default
            :class:`PolynomialPulseMap` represents each pulse parameter as a
            fixed-degree polynomial in the gate parameter.

    """

    detuning_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(0.0))
    phase_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(0.0))
    rabi_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(1.0))
    pulse_map: PulseParamMap = field(default_factory=PolynomialPulseMap)

    @property
    def pulse_ansatz(self) -> PulseAnsatz:
        r"""Generate the pulse ansatz corresponding to a given gate parameter."""
        return PulseAnsatz(
            detuning_ansatz=self.detuning_ansatz, phase_ansatz=self.phase_ansatz, rabi_ansatz=self.rabi_ansatz
        )

    @property
    def param_counts(self) -> tuple[int, int, int]:
        return (
            self.detuning_ansatz.num_params,
            self.phase_ansatz.num_params,
            self.rabi_ansatz.num_params,
        )

    @property
    def shapes(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self.pulse_map.map_shape(self.param_counts)

    @staticmethod
    def target_phase(gate_param: float | jax.Array | None) -> float | jax.Array:
        r"""Return the gate-family parameter normalized by :math:`2\pi`.

        The normalized parameter is used as the input to ``pulse_map`` when
        generating pulse-family parameters.
        """
        if gate_param is None:
            raise TypeError("Expected gate_param to be a float or jax.Array, got None.")
        return gate_param / (2 * np.pi)

    def _unpack_params_arrays(
        self, trainable_params: ParamsFloatLike
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        expected_shapes = self.shapes
        if _is_unpacked(trainable_params):
            duration = jnp.asarray(trainable_params[0])
            detuning_flat = jnp.asarray(trainable_params[1])
            phase_flat = jnp.asarray(trainable_params[2])
            rabi_flat = jnp.asarray(trainable_params[3])
        else:
            flat_params = jnp.ravel(jnp.asarray(trainable_params, dtype=jnp.float64))
            expected_sizes = [prod(t) for t in expected_shapes]
            total_expected_size = sum(expected_sizes)

            if int(flat_params.shape[-1]) != total_expected_size:
                raise ValueError(
                    f"PulseAnsatz expects {total_expected_size} packed parameters, got {int(flat_params.shape[-1])}"
                )

            splits = np.cumsum(expected_sizes[:-1])
            duration, detuning_flat, phase_flat, rabi_flat = jnp.split(flat_params, splits, axis=-1)
        return (
            duration,
            detuning_flat.reshape(expected_shapes[1]),
            phase_flat.reshape(expected_shapes[2]),
            rabi_flat.reshape(expected_shapes[3]),
        )

    def unpack_params(self, trainable_params: ParamsFloatLike) -> PulseFamilyParams[float]:
        r"""Convert pulse-family parameters to a :class:`PulseFamilyParams`.

        Args:
            trainable_params: Packed or unpacked pulse-family trainable parameters.

        Returns:
            Pulse-family duration and ansatz parameter coefficients.

        """
        duration, detuning_params, phase_params, rabi_params = self._unpack_params_arrays(trainable_params)
        return PulseFamilyParams(duration, detuning_params, phase_params, rabi_params)

    def _generate_pulse_params_arrays(
        self, trainable_params: ParamsFloatLike, gate_param: float | jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Evaluate ``pulse_map`` and return generated pulse parameter arrays."""
        unpacked = self._unpack_params_arrays(trainable_params)
        return self.pulse_map.map_full(self.target_phase(gate_param), unpacked)

    def generate_pulse_params(
        self, trainable_params: ParamsFloatLike, gate_param: float | jax.Array | None = None
    ) -> PulseParams:
        r"""Generate duration and ansatz parameter arrays for a gate parameter."""
        duration, detuning_params, phase_params, rabi_params = self._generate_pulse_params_arrays(
            trainable_params, gate_param
        )
        return PulseParams(duration, detuning_params, phase_params, rabi_params)

    def generate_duration(
        self, trainable_params: ParamsFloatLike, gate_param: float | jax.Array | None = None
    ) -> float | jax.Array:
        r"""Generate the pulse duration for a given gate parameter."""
        unpacked = self._unpack_params_arrays(trainable_params)
        return self.pulse_map.map_duration(self.target_phase(gate_param), unpacked)
