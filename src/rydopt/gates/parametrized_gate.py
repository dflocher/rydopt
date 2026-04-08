from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp

from rydopt.protocols import GateSystem, GateWithInterpolationParam, PulseAnsatzLike
from rydopt.simulation import process_fidelity
from rydopt.types import PulseParams


class ParametrizedGate:
    """Collection of gates evaluated under a shared pulse ansatz with interpolation.

    Each gate is paired with a user-defined *interpolation parameter*, which is
    passed to the pulse ansatz (e.g. `MappedPulseAnsatz`) via
    `GateWithInterpolationParam`.

    The fidelity is evaluated independently for each gate and then combined
    according to the specified reduction operation.

    Args:
        gates: Sequence of gate instances defining the physical systems.
        interpolation_params: Sequence of scalar parameters (same length as `gates`)
            that control the pulse parametrization.
        reduction_operation: Reduction operation applied to the per-gate fidelities.
            One of {"mean", "min", "max"}.

    """

    def __init__(
        self,
        gates: Sequence[GateSystem],
        interpolation_params: Sequence[float] | jax.Array,
        reduction_operation: Literal["mean", "min", "max"] = "mean",
    ) -> None:
        if len(gates) != len(interpolation_params):
            raise ValueError("gates and interpolation_params must have same length")

        self.gates = [GateWithInterpolationParam(g, p) for g, p in zip(gates, interpolation_params)]
        self.reduction_operation = reduction_operation

    def fidelity(
        self,
        pulse: PulseAnsatzLike,
        params: PulseParams,
        tol: float,
        fidelity_fn: Callable[
            [GateWithInterpolationParam, PulseAnsatzLike, PulseParams, float], jax.Array
        ] = process_fidelity,
    ) -> jax.Array:
        """Compute reduced fidelity over all parametrized gates.

        Args:
            pulse: Pulse ansatz used for all gates.
            params: Trainable pulse parameters.
            tol: Numerical tolerance passed to the fidelity function.
            fidelity_fn: Function computing fidelity for a single gate.

        Returns:
            Reduced fidelity value according to `self.reduction`.

        """
        fidelities = jnp.stack([fidelity_fn(g, pulse, params, tol) for g in self.gates])

        if self.reduction_operation == "mean":
            return jnp.mean(fidelities)
        if self.reduction_operation == "min":
            return jnp.min(fidelities)
        if self.reduction_operation == "max":
            return jnp.max(fidelities)

        raise ValueError("Invalid reduction")
