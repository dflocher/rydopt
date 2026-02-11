from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp

from rydopt.protocols import GateSystem
from rydopt.pulses import PulseAnsatz
from rydopt.simulation import process_fidelity
from rydopt.types import PulseParams


class ParametrizedGate:
    def __init__(
        self,
        gates: Sequence[GateSystem],
        reduction: Literal["mean", "min", "max"] = "mean",
    ):
        self.gates = gates
        self.reduction = reduction

    def fidelity(
        self,
        pulse: PulseAnsatz,
        params: PulseParams,
        tol: float,
    ) -> jnp.ndarray:
        fidelities = jnp.stack([process_fidelity(g, pulse, params, tol) for g in self.gates])

        if self.reduction == "mean":
            return jnp.mean(fidelities)
        if self.reduction == "min":
            return jnp.min(fidelities)
        if self.reduction == "max":
            return jnp.max(fidelities)

        raise ValueError("Invalid reduction")
