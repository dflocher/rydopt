from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from numpy.typing import ArrayLike

PulseParams = tuple[float, ArrayLike, ArrayLike, ArrayLike]

FixedPulseParams = tuple[bool, ArrayLike, ArrayLike, ArrayLike]

PulseAnsatzFunction = Callable[
    [jnp.ndarray | float, float, jnp.ndarray],
    jnp.ndarray,
]

PulseFunction = Callable[[jnp.ndarray | float], jnp.ndarray]

HamiltonianFunction = Callable[[jnp.ndarray | float, jnp.ndarray | float, jnp.ndarray | float], jnp.ndarray]
