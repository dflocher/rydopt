from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
from numpy.typing import ArrayLike

FidelityType = Literal["process", "average_gate"]

PulseParams = tuple[float, ArrayLike, ArrayLike, ArrayLike]
PulseParamsLike = PulseParams | ArrayLike

FixedPulseParams = tuple[bool, ArrayLike, ArrayLike, ArrayLike]
FixedPulseParamsLike = FixedPulseParams | ArrayLike

PulseFunction = Callable[[float | jax.Array], jax.Array]

HamiltonianFunction = Callable[[float | jax.Array, float | jax.Array, float | jax.Array, float | jax.Array], jax.Array]
