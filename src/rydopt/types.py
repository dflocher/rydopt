from __future__ import annotations

from typing import TypeAlias
from numpy.typing import ArrayLike
from typing import Callable
import jax.numpy as jnp

ParamsTuple: TypeAlias = tuple[float, ArrayLike, ArrayLike, ArrayLike]

FixedParamsTuple: TypeAlias = tuple[bool, ArrayLike, ArrayLike, ArrayLike]

PulseAnsatzFunction = Callable[
    [jnp.ndarray | float, float, jnp.ndarray],
    jnp.ndarray,
]

PulseFunction = Callable[[jnp.ndarray | float], jnp.ndarray]

HamiltonianFunction = Callable[[float, float, float], jnp.ndarray]
