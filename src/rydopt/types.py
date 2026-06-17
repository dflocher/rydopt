from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, cast

import jax
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from rydopt.pulses.pulse_params import PulseParams
else:

    class PulseParams:
        """Runtime stub used to keep type aliases subscriptable without importing pulses."""

        def __class_getitem__(cls, _: object) -> type[PulseParams]:
            """Return the stub class so runtime generic subscripting succeeds."""
            return cast(type[PulseParams], cls)


FidelityType = Literal["process", "average_gate"]
UnpackedParams = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
ParamsFloatLike = PulseParams[float] | Sequence[float] | jax.Array | npt.NDArray[np.float64] | UnpackedParams
ParamsBoolLike = PulseParams[bool] | Sequence[bool] | jax.Array | npt.NDArray[np.bool_]


PulseFunction = Callable[[float | jax.Array], jax.Array]

HamiltonianFunction = Callable[[float | jax.Array, float | jax.Array, float | jax.Array, float | jax.Array], jax.Array]

DurationLike = float | npt.NDArray[np.float64] | jax.Array
