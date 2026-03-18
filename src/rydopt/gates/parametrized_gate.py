from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp

from rydopt.protocols import GateSystem, PulseAnsatzLike
from rydopt.simulation import process_fidelity
from rydopt.types import PulseParams


class ParametrizedGate:
    def __init__(
        self,
        gates: Sequence[GateSystem],
        reduction: Literal["mean", "min", "max"] = "mean",
    ) -> None:
        self.gates = gates
        self.reduction = reduction

    def fidelity(
        self,
        pulse: PulseAnsatzLike,
        params: PulseParams,
        tol: float,
        fidelity_fn: Callable[[GateSystem, PulseAnsatzLike, PulseParams, float], jax.Array] = process_fidelity,
    ) -> jax.Array:
        fidelities = jnp.stack([fidelity_fn(g, pulse, params, tol) for g in self.gates])

        if self.reduction == "mean":
            return jnp.mean(fidelities)
        if self.reduction == "min":
            return jnp.min(fidelities)
        if self.reduction == "max":
            return jnp.max(fidelities)

        raise ValueError("Invalid reduction")
