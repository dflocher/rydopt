from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import jax.numpy as jnp
from numpy.typing import ArrayLike

ParamsTuple: TypeAlias = tuple[float, ArrayLike, ArrayLike, ArrayLike]

FixedParamsTuple: TypeAlias = tuple[bool, ArrayLike, ArrayLike, ArrayLike]

PulseAnsatzFunction = Callable[
    [jnp.ndarray | float, float, jnp.ndarray],
    jnp.ndarray,
]

PulseFunction = Callable[[jnp.ndarray | float], jnp.ndarray]

HamiltonianFunction = Callable[[float, float, float], jnp.ndarray]
