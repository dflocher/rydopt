from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import jax
import jax.numpy as jnp

from rydopt.pulses.general_pulse_ansatz_functions import (
    bspline as _bspline,
    chebyshev as _chebyshev,
    const as _const,
    cos_crab as _cos_crab,
    cos_series as _cos_series,
    cos_sin_crab as _cos_sin_crab,
    legendre as _legendre,
    piecewise_constant as _piecewise_constant,
    polynomial as _polynomial,
    sin_cos_crab as _sin_cos_crab,
    sin_crab as _sin_crab,
    sin_series as _sin_series,
)
from rydopt.pulses.softbox_pulse_ansatz_functions import (
    softbox_blackman as _softbox_blackman,
    softbox_fifth_order_smoothstep as _softbox_fifth_order_smoothstep,
    softbox_hann as _softbox_hann,
    softbox_nuttall as _softbox_nuttall,
    softbox_planck as _softbox_planck,
    softbox_seventh_order_smoothstep as _softbox_seventh_order_smoothstep,
)


class PulseAnsatzFunction(ABC):
    """Abstract base class for configurable pulse ansatz functions."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap subclass call implementations with parameter-size validation."""
        super().__init_subclass__(**kwargs)
        call = cls.__dict__.get("__call__")
        if call is None:
            return

        @wraps(call)
        def validated_call(
            self: PulseAnsatzFunction,
            t: float | jax.Array,
            duration: float | jax.Array,
            ansatz_params: jax.Array,
        ) -> jax.Array:
            validated_params = jnp.asarray(ansatz_params)
            if int(validated_params.size) != self.num_params:
                raise ValueError(
                    f"{type(self).__name__} expects {self.num_params} parameters, got {int(validated_params.size)}"
                )
            return call(self, t, duration, validated_params)

        type.__setattr__(cls, "__call__", validated_call)

    def __init__(self, num_params: int) -> None:
        self._num_params = num_params

    @property
    def num_params(self) -> int:
        """Number of scalar parameters expected by this ansatz."""
        return self._num_params

    @abstractmethod
    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        """Evaluate the ansatz function."""


class Const(PulseAnsatzFunction):
    def __init__(self, num_params: int = 1) -> None:
        if num_params != 1:
            raise ValueError("Const requires exactly 1 parameter")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _const(t, duration, ansatz_params)


class Polynomial(PulseAnsatzFunction):
    def __init__(self, num_params: int = 1) -> None:
        if num_params < 1:
            raise ValueError("Polynomial requires a number of parameters >= 1")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _polynomial(t, duration, ansatz_params)


class SinSeries(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 1:
            raise ValueError("SinSeries requires a number of parameters >= 1")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _sin_series(t, duration, ansatz_params)


class CosSeries(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 1:
            raise ValueError("CosSeries requires a number of parameters >= 1")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _cos_series(t, duration, ansatz_params)


class SinCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 2 or num_params % 2 != 0:
            raise ValueError("SinCrab requires an even number of parameters >= 2")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _sin_crab(t, duration, ansatz_params)


class CosCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 2 or num_params % 2 != 0:
            raise ValueError("CosCrab requires an even number of parameters >= 2")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _cos_crab(t, duration, ansatz_params)


class SinCosCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 4 or num_params % 4 != 0:
            raise ValueError("SinCosCrab requires a parameter count divisible by 4 and >= 4")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _sin_cos_crab(t, duration, ansatz_params)


class CosSinCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 4 or num_params % 4 != 0:
            raise ValueError("CosSinCrab requires a parameter count divisible by 4 and >= 4")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _cos_sin_crab(t, duration, ansatz_params)


class Chebyshev(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 1:
            raise ValueError("ChebyshevPulse requires at least one parameter")
        super().__init__(num_params)

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:
        return _chebyshev(t, duration, ansatz_params)


class Legendre(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 1:
            raise ValueError("LegendrePulse requires at least one parameter")
        super().__init__(num_params)

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:
        return _legendre(t, duration, ansatz_params)


class BSpline(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 4:
            raise ValueError("BSplinePulse requires at least four parameters")
        super().__init__(num_params)

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:
        return _bspline(t, duration, ansatz_params)


class PiecewiseConstant(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 1:
            raise ValueError("PiecewiseConstant requires at least one parameter")
        super().__init__(num_params)

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:
        return _piecewise_constant(t, duration, ansatz_params)


class SoftBoxHann(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxHann requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_hann(t, duration, ansatz_params)


class SoftBoxBlackman(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxBlackman requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_blackman(t, duration, ansatz_params)


class SoftBoxNuttall(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxNuttall requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_nuttall(t, duration, ansatz_params)


class SoftBoxPlanck(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxPlanck requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_planck(t, duration, ansatz_params)


class SoftBoxFifthOrderSmoothstep(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxFifthOrderSmoothstep requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_fifth_order_smoothstep(t, duration, ansatz_params)


class SoftBoxSeventhOrderSmoothstep(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxSeventhOrderSmoothstep requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: float | jax.Array, duration: float | jax.Array, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_seventh_order_smoothstep(t, duration, ansatz_params)


class Symmetric(PulseAnsatzFunction):
    r"""Time-symmetric pulse ansatz.

    Constructs a symmetric pulse from a base pulse ansatz :math:`g(t)` by
    reflecting it about the midpoint of the pulse duration,

    .. math::

       f(t)
       =
       \frac{1}{2}
       \left[
           g(t)
           +
           g(T-t)
       \right].

    The resulting pulse satisfies

    .. math::

       f(t)=f(T-t).

    Args:
        base_ansatz: Pulse ansatz to be symmetrized.

    """

    def __init__(self, base_ansatz: PulseAnsatzFunction) -> None:
        super().__init__(base_ansatz.num_params)
        self._base_ansatz = base_ansatz

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:
        pulse = self._base_ansatz(t, duration, ansatz_params)
        mirrored = self._base_ansatz(duration - t, duration, ansatz_params)
        return 0.5 * (pulse + mirrored)


class AntiSymmetric(PulseAnsatzFunction):
    r"""Time-antisymmetric pulse ansatz.

    Constructs an antisymmetric pulse from a base pulse ansatz :math:`g(t)` by
    reflecting it about the midpoint of the pulse duration,

    .. math::

       f(t)
       =
       \frac{1}{2}
       \left[
           g(t)
           -
           g(T-t)
       \right].

    The resulting pulse satisfies

    .. math::

       f(t)=-f(T-t).

    Args:
        base_ansatz: Pulse ansatz to be antisymmetrized.

    """

    def __init__(self, base_ansatz: PulseAnsatzFunction) -> None:
        super().__init__(base_ansatz.num_params)
        self._base_ansatz = base_ansatz

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:
        pulse = self._base_ansatz(t, duration, ansatz_params)
        mirrored = self._base_ansatz(duration - t, duration, ansatz_params)
        return 0.5 * (pulse - mirrored)


class Shifted(PulseAnsatzFunction):
    r"""Vertically shifted pulse ansatz.

    Constructs a pulse from a base pulse ansatz :math:`g(t)` by adding a
    constant offset,

    .. math::

       f(t)
       =
       g(t)
       +
       c.

    Args:
        base_ansatz: Pulse ansatz to be shifted.

    """

    def __init__(self, base_ansatz: PulseAnsatzFunction) -> None:
        super().__init__(base_ansatz.num_params + 1)
        self._base_ansatz = base_ansatz

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:
        offset = ansatz_params[0]
        pulse = self._base_ansatz(t, duration, ansatz_params[1:])
        return offset + pulse


class Product(PulseAnsatzFunction):
    r"""Product of two pulse ansatz functions.

    Constructs a pulse from two pulse ansatz functions :math:`g_1(t)` and
    :math:`g_2(t)`,

    .. math::

       f(t)
       =
       g_1(t)
       g_2(t).

    This wrapper is particularly useful for constructing envelope-modulated
    pulses, where one pulse ansatz represents an envelope and the other a
    carrier.

    Args:
        pulse1: First pulse ansatz.
        pulse2: Second pulse ansatz.

    """

    def __init__(
        self,
        pulse1: PulseAnsatzFunction,
        pulse2: PulseAnsatzFunction,
    ) -> None:
        super().__init__(pulse1.num_params + pulse2.num_params)
        self._pulse1 = pulse1
        self._pulse2 = pulse2

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:

        n = self._pulse1.num_params
        pulse1 = self._pulse1(t, duration, ansatz_params[:n])
        pulse2 = self._pulse2(t, duration, ansatz_params[n:])

        return pulse1 * pulse2


class Sum(PulseAnsatzFunction):
    r"""Sum of two pulse ansatz functions.

    Constructs a pulse from two pulse ansatz functions :math:`g_1(t)` and
    :math:`g_2(t)`,

    .. math::

       f(t)
       =
       g_1(t)
       +
       g_2(t).

    Args:
        pulse1: First pulse ansatz.
        pulse2: Second pulse ansatz.

    """

    def __init__(
        self,
        pulse1: PulseAnsatzFunction,
        pulse2: PulseAnsatzFunction,
    ) -> None:
        super().__init__(pulse1.num_params + pulse2.num_params)
        self._pulse1 = pulse1
        self._pulse2 = pulse2

    def __call__(
        self,
        t: float | jax.Array,
        duration: float | jax.Array,
        ansatz_params: jax.Array,
    ) -> jax.Array:

        n = self._pulse1.num_params

        pulse1 = self._pulse1(t, duration, ansatz_params[:n])
        pulse2 = self._pulse2(t, duration, ansatz_params[n:])

        return pulse1 + pulse2
