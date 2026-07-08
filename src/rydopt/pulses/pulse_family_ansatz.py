from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from rydopt.pulses.pulse_ansatz import PulseAnsatzFunction, SinglePhotonPulseAnsatz, _FixedConstant, _is_unpacked
from rydopt.pulses.pulse_family_params import PulseFamilyParams
from rydopt.pulses.pulse_params import PulseParams
from rydopt.types import ParamsFloatLike


@runtime_checkable
class PulseParamMap(Protocol):
    """Minimal interface for the map of ansatz parameters used in PulseFamilyAnsatz."""

    def map_duration(
        self,
        target_argument: float | jax.Array,
        packed_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> float | jax.Array: ...

    def map_full(
        self,
        target_argument: float | jax.Array,
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
    target argument of the gate. Each component is treated as a polynomial in the
    target argument, with per-component degree given by ``degrees``.

    Args:
        degrees: polynomial degree for
            ``(duration, detuning, phase, rabi)``.

    """

    degrees: Sequence[int] = field(default_factory=lambda: [0, 0, 0, 0])

    @staticmethod
    def _poly_eval(target_argument: jax.Array, coeffs: jax.Array, degree: int) -> jax.Array:
        """Evaluate a polynomial in ``target_argument``.

        Works for 1-D ``coeffs`` of shape ``(degree + 1,)`` returning a scalar,
        and for 2-D ``coeffs`` of shape ``(n, degree + 1)`` returning shape ``(n,)``.
        """
        powers = jnp.power(target_argument / (2 * jnp.pi), jnp.arange(degree + 1))
        return coeffs @ powers

    def map_duration(
        self,
        target_argument: float | jax.Array,
        packed_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        target_argument = jnp.asarray(target_argument)
        params = jnp.ravel(jnp.asarray(packed_params[0]))
        degree = self.degrees[0]

        if degree > 0:
            duration = self._poly_eval(target_argument, params, degree)
            return jax.nn.softplus(duration)
        return params[0]

    def map_full(
        self,
        target_argument: float | jax.Array,
        packed_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        target_argument = jnp.asarray(target_argument)

        # --- Duration ---
        duration = self.map_duration(target_argument, packed_params)

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
            outputs.append(self._poly_eval(target_argument, coeffs, degree))

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


def empirical_cphase_duration(target_phase: jax.Array, duration_params: jax.Array) -> jax.Array:
    r"""Empirical expression for the duration of controlled-phase gates.

    The duration parameters are ``(piduration, prefactor, exponent)``. The
    expression is based on fitting the functional form to controlled-phase
    gate durations for a range of target phases. The durations were extracted
    from Extended Data Fig. 5b of `Evered et al., Nature 622, 268-272 (2023)
    <https://doi.org/10.1038/s41586-023-06481-y>`_. The expression is given by

    .. math::

        T(\theta) =
        T_\pi
        \left[
        1 - \left(1 - x^p\right)^q
        \right],

    where

    .. math::

        x = 1 - \left|\frac{\theta}{\pi} - 1\right|,
        \qquad
        q = \frac{A}{T_\pi 2^p}.

    Here :math:`\theta` is the target phase in radians clipped to
    the range :math:`[0, 2\pi]`, :math:`T_\pi`
    is ``piduration``, :math:`A` is ``prefactor``, and :math:`p` is
    ``exponent``. The expression is symmetric under
    :math:`\theta \mapsto 2\pi - \theta`, satisfies
    :math:`T(\pi) = T_\pi`, and has the endpoint behavior

    .. math::

         T(\theta) \sim A\left(\frac{\theta}{2\pi}\right)^p
         \qquad \theta \to 0,

    with the corresponding symmetric behavior near :math:`\theta \to 2\pi`.

    Args:
        target_phase: Target phase :math:`\theta` in radians.
        duration_params: Array of the three duration parameters
            ``(piduration, prefactor, exponent)``.

    Returns:
        The pulse duration :math:`T(\theta)`.

    """
    if duration_params.size != 3:
        raise ValueError(
            "empirical_cphase_duration expects three duration parameters: (piduration, prefactor, exponent)"
        )

    piduration, prefactor, exponent = duration_params

    phase_over_2pi = jnp.clip(target_phase / (2.0 * jnp.pi), 0.0, 1.0)
    x = 2.0 * jnp.minimum(phase_over_2pi, 1.0 - phase_over_2pi)
    q = prefactor / (piduration * 2.0**exponent)
    safe_inner = jnp.maximum(1.0 - x**exponent, jnp.finfo(x.dtype).tiny)
    duration = piduration * (-jnp.expm1(q * jnp.log(safe_inner)))
    return jnp.where(x >= 1.0, piduration, duration)


@dataclass
class PolynomialPulseMapWithCustomDuration(PolynomialPulseMap):
    r"""Polynomial map of ansatz parameters with a custom expression for the gate duration.

    The duration is computed by a user-provided callable ``duration_map`` that
    takes the target argument and the duration parameter array and returns the
    pulse duration. By default, the empirical expression
    :func:`empirical_cphase_duration` with the three duration parameters
    ``(piduration, prefactor, exponent)`` is used. Detuning, laser
    phase, and Rabi parameters use the polynomial mapping from
    :class:`PolynomialPulseMap`.

    Args:
        degrees: polynomial degree for
            ``(duration, detuning, phase, rabi)``. The duration degree is
            ignored because the duration uses ``duration_map``.
        duration_map: Callable ``(target_argument, duration_params) -> duration``
            used to compute the pulse duration. Defaults to
            :func:`empirical_cphase_duration`.
        num_duration_params: Number of duration parameters expected by
            ``duration_map``. Defaults to ``3``, matching
            :func:`empirical_cphase_duration`.

    """

    duration_map: Callable[[jax.Array, jax.Array], jax.Array] = empirical_cphase_duration
    num_duration_params: int = 3

    def map_duration(
        self,
        target_argument: float | jax.Array,
        packed_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        target = jnp.asarray(target_argument)
        params = jnp.ravel(jnp.asarray(packed_params[0]))

        if params.size != self.num_duration_params:
            raise ValueError(
                f"PolynomialPulseMapWithCustomDuration expects {self.num_duration_params} "
                f"duration parameters, got {params.size}"
            )

        return self.duration_map(target, params)

    def map_shape(
        self,
        params_count: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (
            (self.num_duration_params,),
            self.shape(params_count[0], self.degrees[1]),
            self.shape(params_count[1], self.degrees[2]),
            self.shape(params_count[2], self.degrees[3]),
        )


@dataclass
class PulseFamilyAnsatz:
    r"""Data class that stores ansatz functions for a family of laser pulses.

    A pulse family describes a continuous family of gates argumentized by a gate argument
    :math:`\phi`
    (for example, the target phase of a controlled phase gate). Rather than optimizing an independent pulse
    for each value of :math:`\phi`, the pulse duration and ansatz parameters are represented as functions of
    :math:`\phi`.

    RydOpt models this dependence through a parameter map. The packed pulse parameters are optimized once
    and mapped to the pulse duration and ansatz parameters for a specific gate argument value.
    By default, :class:`PolynomialPulseMap` represents each pulse parameter as a polynomial of fixed degree
    in :math:`\phi`.

    For available ansatz functions for the detuning :math:`\Delta(t)`, phase :math:`\xi(t)`,
    and Rabi frequency :math:`\Omega(t)` sweeps, see below. The function :func:`optimize_family
    <rydopt.optimization.optimize>` allows optimizing the pulse-family parameters to maximize
    fidelity across a target gate family. Initial pulse-family parameters can be provided as
    ``PulseFamilyParams(duration_params, detuning_params, phase_params, rabi_params)``,
    where each array contains the coefficients used by ``pulse_map`` to construct the corresponding
    pulse duration or ansatz parameters for a given gate argument value.

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
            ansatz parameters for a given gate argument value. The default
            :class:`PolynomialPulseMap` represents each pulse parameter as a
            fixed-degree polynomial in the gate argument.

    """

    detuning_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(0.0))
    phase_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(0.0))
    rabi_ansatz: PulseAnsatzFunction = field(default_factory=lambda: _FixedConstant(1.0))
    pulse_map: PulseParamMap = field(default_factory=PolynomialPulseMap)

    @property
    def pulse_ansatz(self) -> SinglePhotonPulseAnsatz:
        r"""Generate the pulse ansatz corresponding to a given gate argument."""
        return SinglePhotonPulseAnsatz(
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
    def target_argument(gate_arg: float | jax.Array | None) -> float | jax.Array:
        r"""Return the gate-family argument.

        The argument is used as the input to ``pulse_map`` when
        generating pulse-family parameters.
        """
        if gate_arg is None:
            raise TypeError("Expected gate_arg to be a float or jax.Array, got None.")
        return gate_arg

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
                    f"SinglePhotonPulseAnsatz expects {total_expected_size} packed parameters, "
                    f"got {int(flat_params.shape[-1])}"
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
        self, trainable_params: ParamsFloatLike, gate_arg: float | jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Evaluate ``pulse_map`` and return generated pulse parameter arrays."""
        unpacked = self._unpack_params_arrays(trainable_params)
        return self.pulse_map.map_full(self.target_argument(gate_arg), unpacked)

    def generate_pulse_params(
        self, trainable_params: ParamsFloatLike, gate_arg: float | jax.Array | None = None
    ) -> PulseParams:
        r"""Generate duration and ansatz parameter arrays for a gate argument."""
        duration, detuning_params, phase_params, rabi_params = self._generate_pulse_params_arrays(
            trainable_params, gate_arg
        )
        return PulseParams(duration, detuning_params, phase_params, rabi_params)

    def generate_duration(
        self, trainable_params: ParamsFloatLike, gate_arg: float | jax.Array | None = None
    ) -> float | jax.Array:
        r"""Generate the pulse duration for a given gate argument."""
        unpacked = self._unpack_params_arrays(trainable_params)
        return self.pulse_map.map_duration(self.target_argument(gate_arg), unpacked)
