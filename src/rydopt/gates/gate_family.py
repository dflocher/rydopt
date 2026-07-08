import math
from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from rydopt.protocols import GateSystem
from rydopt.pulses import PulseFamilyAnsatz
from rydopt.types import OneDimensionalArrayLike, ParamsFloatLike


class GateFamily:
    """Collection of gates evaluated under a shared pulse ansatz with interpolation.

    The infidelity is evaluated independently for each gate and then combined
    according to the specified reduction operation.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> target_phases = np.linspace(0.25, 1.0, 4) * np.pi
        >>> sampled_gates = [
        ...     ro.gates.TwoQubitGate(
        ...         phi=None,
        ...         theta=np.pi,
        ...         Vnn=float("inf"),
        ...         decay=0.0001,
        ...     )
        ...     for phase in target_phases
        ... ]
        >>> parametrized_gate = ro.gates.GateFamily(
        ...     fixed_parameter_gates=sampled_gates,
        ...     parameter_values=target_phases,
        ...     reduction="mean",
        ... )

    Args:
        fixed_parameter_gates: Sequence of gate instances defining the physical systems.
        parameter_values: Sequence of scalar parameters (same length as `fixed_parameter_gates`)
            that controls the pulse family parametrization.
        reduction: Reduction operation applied to the per-gate infidelities.
            One of {"mean", "max", "softmax"}.
        softmax_scale: Non-negative scale parameter used only when `reduction="softmax"`.
            A value of zero corresponds to the max, and infinity corresponds to the mean.
            Finite values set the scale of per-gate infidelity differences that the
            optimizer should care about.

    """

    def __init__(
        self,
        fixed_parameter_gates: Sequence[GateSystem],
        parameter_values: OneDimensionalArrayLike,
        reduction: Literal["mean", "max", "softmax"] = "mean",
        softmax_scale: float | None = None,
    ) -> None:
        self.parameter_values = jnp.asarray(parameter_values, dtype=np.float64)
        self.gates = list(fixed_parameter_gates)

        if len(fixed_parameter_gates) != len(self.parameter_values):
            raise ValueError("fixed_parameter_gates and parameter_values must have the same length.")

        self._num_gates = len(fixed_parameter_gates)

        if reduction == "mean":
            if softmax_scale is not None:
                raise ValueError("softmax_scale may only be provided when reduction='softmax'.")
            self.reduction = float("inf")
        elif reduction == "max":
            if softmax_scale is not None:
                raise ValueError("softmax_scale may only be provided when reduction='softmax'.")
            self.reduction = 0.0
        elif reduction == "softmax":
            if softmax_scale is None:
                raise ValueError("softmax_scale must be provided when reduction='softmax'.")
            self.reduction = softmax_scale
        else:
            raise ValueError("Invalid reduction, must be 'mean', 'max', or 'softmax'.")

    def cost(self, pulse: PulseFamilyAnsatz, params: ParamsFloatLike, tol: float) -> jax.Array:
        """Compute reduced infidelity over all fixed-target-parameter gates defined within the
        gate family.

        Args:
            pulse: Pulse family ansatz used for all gates.
            params: Trainable pulse family parameters.
            tol: Numerical tolerance passed to the cost function.

        Returns:
            Reduced infidelity value according to `self.reduction`.

        """
        pulse_ansatz = pulse.pulse_ansatz
        costs = jnp.stack(
            [
                gate.cost(pulse_ansatz, pulse.generate_pulse_params(params, pv), tol)
                for gate, pv in zip(self.gates, self.parameter_values)
            ]
        )
        if self.reduction == 0.0:
            return jnp.max(costs)

        if math.isinf(self.reduction):
            return jnp.mean(costs)

        return self.reduction * (logsumexp(costs / self.reduction) - math.log(self._num_gates))
