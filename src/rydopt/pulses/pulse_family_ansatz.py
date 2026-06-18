from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from rydopt.protocols import PulseAnsatzLike
from rydopt.pulses.pulse_ansatz import PulseAnsatzFunction, _FixedConstant, _is_unpacked
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
    r"""Stores ansatz functions for a pulse family.

    The parameters of the ansatz functions and the pulse duration are optimized
    to maximize gate fidelity. Parameters are treated as fixed-degree
    polynomials in the parametrized target phase of the gate.

    Example:
        >>> import rydopt as ro
        >>> degrees = [0, 0, 3, 0]
        >>> num_phase_params = 10
        >>> pulse_map = ro.pulses.PolynomialPulseMap(degrees)
        >>> pulse_family = ro.pulses.PulseFamilyAnsatz(
        ...     detuning_ansatz=ro.pulses.Const(),
        ...     phase_ansatz=ro.pulses.SinCrab(num_phase_params),
        ...     pulse_map=pulse_map,
        ... )

    Attributes:
        detuning_ansatz: Detuning sweep, default is zero.
        phase_ansatz: Phase sweep, default is zero.
        rabi_ansatz: Rabi frequency amplitude sweep, default is one.
        pulse_map: Map from packed pulse parameters to ansatz parameters given a
            target phase, default is :class:`PolynomialPulseMap`.

    """

    detuning_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(0.0))
    phase_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(0.0))
    rabi_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(1.0))
    pulse_map: PulseParamMap = field(default_factory=PolynomialPulseMap)

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
        if gate_param is None:
            raise TypeError("Expected gate_param to be a float or jax.Array, got None.")
        return gate_param / (2 * np.pi)

    def _unpack_params_arrays(
        self, trainable_params: ParamsFloatLike
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if _is_unpacked(trainable_params):
            return tuple(trainable_params)  # type: ignore[return-value]
        flat_params = jnp.ravel(jnp.asarray(trainable_params, dtype=jnp.float64))
        expected_shapes = self.shapes
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
        duration, detuning_params, phase_params, rabi_params = self._unpack_params_arrays(trainable_params)
        return PulseFamilyParams(duration, detuning_params, phase_params, rabi_params)

    def _generate_pulse_params_arrays(
        self, trainable_params: ParamsFloatLike, gate_param: float | jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        unpacked = self._unpack_params_arrays(trainable_params)
        return self.pulse_map.map_full(self.target_phase(gate_param), unpacked)

    def generate_pulse_params(
        self, trainable_params: ParamsFloatLike, gate_param: float | jax.Array | None = None
    ) -> PulseParams:
        duration, detuning_params, phase_params, rabi_params = self._generate_pulse_params_arrays(
            trainable_params, gate_param
        )
        return PulseParams(float(duration), detuning_params, phase_params, rabi_params)

    def generate_duration(
        self, trainable_params: ParamsFloatLike, gate_param: float | jax.Array | None = None
    ) -> float | jax.Array:
        unpacked = self._unpack_params_arrays(trainable_params)
        return self.pulse_map.map_duration(self.target_phase(gate_param), unpacked)

    def evaluate_pulse_functions(
        self,
        t: jax.Array | float,
        params: ParamsFloatLike,
        gate_param: float | jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Evaluate the detuning, phase, and rabi sweeps at the given times for a
        gate family with a predefined target phase.

        Args:
            t: time samples at which the functions are evaluated.
            params: family pulse parameters.
            gate_param: parameter value corresponding to a predefined gate instance.

        Returns:
            Tuple ``(detuning_1, detuning_r, phase, rabi)``.

        """
        (
            duration,
            detuning_ansatz_params,
            phase_ansatz_params,
            rabi_ansatz_params,
        ) = self._generate_pulse_params_arrays(params, gate_param)
        return (
            jnp.zeros_like(t),
            self.detuning_ansatz(t, duration, detuning_ansatz_params),
            self.phase_ansatz(t, duration, phase_ansatz_params),
            self.rabi_ansatz(t, duration, rabi_ansatz_params),
        )

    def generate_pulse_ansatz(self, gate_param: float | jax.Array | None = None) -> PulseAnsatzLike:
        return BoundPulseAnsatz(self, gate_param=gate_param)


@jax.tree_util.register_pytree_node_class
@dataclass
class BoundPulseAnsatz:
    family: PulseFamilyAnsatz
    gate_param: float | jax.Array | None

    def evaluate_pulse_functions(
        self, t: float | jax.Array, params: ParamsFloatLike, gate_param: float | jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        del gate_param
        (
            duration,
            detuning_ansatz_params,
            phase_ansatz_params,
            rabi_ansatz_params,
        ) = params
        return (
            jnp.zeros_like(t),
            self.family.detuning_ansatz(t, duration, jnp.asarray(detuning_ansatz_params)),
            self.family.phase_ansatz(t, duration, jnp.asarray(phase_ansatz_params)),
            self.family.rabi_ansatz(t, duration, jnp.asarray(rabi_ansatz_params)),
        )

    def unpack_params(self, trainable_params: ParamsFloatLike) -> PulseParams[float]:
        return self.family.generate_pulse_params(trainable_params, self.gate_param)

    # --- pytree protocol ---
    def tree_flatten(self) -> tuple[tuple[float | jax.Array | None], PulseFamilyAnsatz]:
        # gate_param is a (traceable) leaf; family is static aux data.
        return (self.gate_param,), self.family

    @classmethod
    def tree_unflatten(cls, aux_data: PulseFamilyAnsatz, children: tuple[float | jax.Array | None]) -> PulseAnsatzLike:
        (gate_param,) = children
        return cls(family=aux_data, gate_param=gate_param)
