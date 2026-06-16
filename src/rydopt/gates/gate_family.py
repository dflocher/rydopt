from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp

from rydopt.protocols import RydbergSystem
from rydopt.pulses import PulseFamilyAnsatz
from rydopt.types import ParamsFloatLike

_REDUCTIONS = {"mean", "max"}
_REDUCTION_FNS = {
    "mean": jnp.mean,
    "max": jnp.max,
}


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
            One of {"mean", "max"}.

    """

    def __init__(
        self,
        fixed_parameter_gates: Sequence[RydbergSystem],
        parameter_values: Sequence[float] | jax.Array,
        reduction: Literal["mean", "max"] = "mean",
    ) -> None:
        if len(fixed_parameter_gates) != len(parameter_values):
            raise ValueError("fixed_parameter_gates and parameter_values must have the same length.")

        self.gates = list(fixed_parameter_gates)
        self.parameter_values = [float(p) for p in parameter_values]
        self._num_gates = len(fixed_parameter_gates)
        if reduction in _REDUCTIONS:
            self.reduction = reduction
        else:
            raise ValueError("Invalid reduction, must be mean or max.")

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
        costs = jnp.stack(
            [
                gate.cost(pulse.generate_pulse_ansatz(pv), params, tol)
                for gate, pv in zip(self.gates, self.parameter_values)
            ]
        )

        return _REDUCTION_FNS[self.reduction](costs)
