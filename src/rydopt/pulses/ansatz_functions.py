from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import jax
import jax.numpy as jnp

from rydopt.pulses.general_pulse_ansatz_functions import (
    const as _const,
    const_cos_crab as _const_cos_crab,
    const_cos_sin_crab as _const_cos_sin_crab,
    const_sin_cos_crab as _const_sin_cos_crab,
    const_sin_crab as _const_sin_crab,
    cos_crab as _cos_crab,
    cos_sin_crab as _cos_sin_crab,
    lin_cos_crab as _lin_cos_crab,
    lin_cos_sin_crab as _lin_cos_sin_crab,
    lin_sin_cos_crab as _lin_sin_cos_crab,
    lin_sin_crab as _lin_sin_crab,
    sin_cos_crab as _sin_cos_crab,
    sin_crab as _sin_crab,
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
    """Base class for configurable pulse ansatz functions."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap subclass call implementations with parameter-size validation."""
        super().__init_subclass__(**kwargs)
        call = cls.__dict__.get("__call__")
        if call is None:
            return

        @wraps(call)
        def validated_call(
            self: PulseAnsatzFunction, t: jax.Array | float, duration: float, ansatz_params: jax.Array
        ) -> jax.Array:
            validated_params = jnp.asarray(ansatz_params)
            if int(validated_params.size) != self.num_params:
                raise ValueError(
                    f"{type(self).__name__} expects {self.num_params} parameters, got {int(validated_params.size)}"
                )
            return call(self, t, duration, validated_params)

        cls.__call__ = validated_call  # type: ignore[invalid-assignment]

    def __init__(self, num_params: int) -> None:
        self._num_params = num_params

    @property
    def num_params(self) -> int:
        """Number of scalar parameters expected by this ansatz."""
        return self._num_params

    @abstractmethod
    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        """Evaluate the ansatz function."""


class Const(PulseAnsatzFunction):
    def __init__(self, num_params: int = 1) -> None:
        if num_params != 1:
            raise ValueError("Const requires exactly 1 parameter")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _const(t, duration, ansatz_params)


class SinCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 2 or num_params % 2 != 0:
            raise ValueError("SinCrab requires an even number of parameters >= 2")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _sin_crab(t, duration, ansatz_params)


class CosCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 2 or num_params % 2 != 0:
            raise ValueError("CosCrab requires an even number of parameters >= 2")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _cos_crab(t, duration, ansatz_params)


class SinCosCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 4 or num_params % 4 != 0:
            raise ValueError("SinCosCrab requires a parameter count divisible by 4 and >= 4")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _sin_cos_crab(t, duration, ansatz_params)


class CosSinCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 4 or num_params % 4 != 0:
            raise ValueError("CosSinCrab requires a parameter count divisible by 4 and >= 4")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _cos_sin_crab(t, duration, ansatz_params)


class ConstSinCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 3 or num_params % 2 == 0:
            raise ValueError("ConstSinCrab requires an odd number of parameters >= 3")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _const_sin_crab(t, duration, ansatz_params)


class ConstCosCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 3 or num_params % 2 == 0:
            raise ValueError("ConstCosCrab requires an odd number of parameters >= 3")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _const_cos_crab(t, duration, ansatz_params)


class ConstSinCosCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 5 or (num_params - 1) % 4 != 0:
            raise ValueError("ConstSinCosCrab requires a parameter count of 4n+1 with n >= 1")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _const_sin_cos_crab(t, duration, ansatz_params)


class ConstCosSinCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 5 or (num_params - 1) % 4 != 0:
            raise ValueError("ConstCosSinCrab requires a parameter count of 4n+1 with n >= 1")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _const_cos_sin_crab(t, duration, ansatz_params)


class LinSinCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 3 or num_params % 2 == 0:
            raise ValueError("LinSinCrab requires an odd number of parameters >= 3")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _lin_sin_crab(t, duration, ansatz_params)


class LinCosCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 3 or num_params % 2 == 0:
            raise ValueError("LinCosCrab requires an odd number of parameters >= 3")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _lin_cos_crab(t, duration, ansatz_params)


class LinSinCosCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 5 or (num_params - 1) % 4 != 0:
            raise ValueError("LinSinCosCrab requires a parameter count of 4n+1 with n >= 1")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _lin_sin_cos_crab(t, duration, ansatz_params)


class LinCosSinCrab(PulseAnsatzFunction):
    def __init__(self, num_params: int) -> None:
        if num_params < 5 or (num_params - 1) % 4 != 0:
            raise ValueError("LinCosSinCrab requires a parameter count of 4n+1 with n >= 1")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _lin_cos_sin_crab(t, duration, ansatz_params)


class SoftBoxHann(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxHann requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_hann(t, duration, ansatz_params)


class SoftBoxBlackman(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxBlackman requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_blackman(t, duration, ansatz_params)


class SoftBoxNuttall(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxNuttall requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_nuttall(t, duration, ansatz_params)


class SoftBoxPlanck(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxPlanck requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_planck(t, duration, ansatz_params)


class SoftBoxFifthOrderSmoothstep(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxFifthOrderSmoothstep requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_fifth_order_smoothstep(t, duration, ansatz_params)


class SoftBoxSeventhOrderSmoothstep(PulseAnsatzFunction):
    def __init__(self, num_params: int = 2) -> None:
        if num_params != 2:
            raise ValueError("SoftBoxSeventhOrderSmoothstep requires exactly 2 parameters")
        super().__init__(num_params)

    def __call__(self, t: jax.Array | float, duration: float, ansatz_params: jax.Array) -> jax.Array:
        return _softbox_seventh_order_smoothstep(t, duration, ansatz_params)
