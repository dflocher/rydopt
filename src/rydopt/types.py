from __future__ import annotations

from collections.abc import Callable

import jax
from numpy.typing import ArrayLike

PulseParams = tuple[float, ArrayLike, ArrayLike, ArrayLike]
GenericPulseParams = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
FixedPulseParams = tuple[bool, ArrayLike, ArrayLike, ArrayLike]

PulseAnsatzFunction = Callable[
    [jax.Array | float, jax.Array | float, jax.Array],
    jax.Array,
]

PulseFunction = Callable[[jax.Array | float], jax.Array]

HamiltonianFunction = Callable[[jax.Array | float, jax.Array | float, jax.Array | float, jax.Array | float], jax.Array]
