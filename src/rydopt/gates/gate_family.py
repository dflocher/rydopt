import math
from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from rydopt.protocols import GateSystem
from rydopt.pulses import PulseFamilyAnsatz
from rydopt.types import ParamsFloatLike


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
            One of {"mean", "max"} or a float value for a softmax reduction (a value of 0.0
            corresponds to the max, and a value of infinity corresponds to the mean;
            finite values set the scale of per-gate infidelity differences that the
            optimizer should care about).

    """

    def __init__(
        self,
        fixed_parameter_gates: Sequence[GateSystem],
        parameter_values: Sequence[float] | jax.Array,
        reduction: Literal["mean", "max"] | float = "mean",
    ) -> None:
        if len(fixed_parameter_gates) != len(parameter_values):
            raise ValueError("fixed_parameter_gates and parameter_values must have the same length.")

        self.gates = list(fixed_parameter_gates)
        self.parameter_values = [float(p) for p in parameter_values]
        self._num_gates = len(fixed_parameter_gates)
        if isinstance(reduction, float) and reduction >= 0.0:
            self.reduction = reduction
        elif reduction == "mean":
            self.reduction = float("inf")
        elif reduction == "max":
            self.reduction = 0.0
        else:
            raise ValueError("Invalid reduction, must be 'mean', 'max', or a non-negative float.")

    def cost(self, pulse: PulseFamilyAnsatz, params: ParamsFloatLike, tol: float) -> jax.Array:
        """Compute reduced infidelity over all fixed-target-phase gates defined within the
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
