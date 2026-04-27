from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
from numpy.typing import ArrayLike

FidelityType = Literal["process", "average_gate"]

PulseParams = tuple[float, ArrayLike, ArrayLike, ArrayLike]

FixedPulseParams = tuple[bool, ArrayLike, ArrayLike, ArrayLike]

PulseFunction = Callable[[jax.Array | float], jax.Array]

HamiltonianFunction = Callable[[jax.Array | float, jax.Array | float, jax.Array | float, jax.Array | float], jax.Array]
