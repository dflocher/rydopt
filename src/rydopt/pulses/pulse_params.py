from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.typing import ArrayLike as JaxArrayLike

ParamScalar = TypeVar("ParamScalar", float, bool)


@jax.tree_util.register_pytree_node_class
class PulseParams(Sequence[jax.Array], Generic[ParamScalar]):
    r"""Pulse-parameter container.

    The container stores pulse parameters components
    ``(duration, detuning_params, phase_params, rabi_params)``.
    """

    __slots__ = ("_detuning_params", "_duration", "_phase_params", "_rabi_params")

    def __init__(
        self,
        duration: JaxArrayLike | npt.ArrayLike,
        detuning_params: JaxArrayLike | npt.ArrayLike = (),
        phase_params: JaxArrayLike | npt.ArrayLike = (),
        rabi_params: JaxArrayLike | npt.ArrayLike = (),
    ) -> None:
        self._duration = jnp.asarray(duration).reshape(1)
        self._detuning_params = jnp.asarray(detuning_params).reshape(-1)
        self._phase_params = jnp.asarray(phase_params).reshape(-1)
        self._rabi_params = jnp.asarray(rabi_params).reshape(-1)

    def __len__(self) -> int:
        """Return the number of parameter components."""
        return 4

    @property
    def duration(self) -> jax.Array:
        return jnp.asarray(self._duration)

    @property
    def detuning_params(self) -> jax.Array:
        return jnp.asarray(self._detuning_params)

    @property
    def phase_params(self) -> jax.Array:
        return jnp.asarray(self._phase_params)

    @property
    def rabi_params(self) -> jax.Array:
        return jnp.asarray(self._rabi_params)

    @property
    def _components(
        self,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return (
            self._duration,
            self._detuning_params,
            self._phase_params,
            self._rabi_params,
        )

    @overload
    def __getitem__(self, index: int) -> jax.Array: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[jax.Array]: ...

    def __getitem__(self, index: int | slice) -> jax.Array | Sequence[jax.Array]:
        """Return one parameter component or a sliced tuple of parameter components."""
        if not isinstance(index, slice) and index == 0:
            return self._duration[0]
        return self._components[index]

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_]:
        """Return the flattened representation used by ``np.asarray``."""
        array = np.concatenate(self._components)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            return array.copy()

        return array

    def __jax_array__(self) -> jax.Array:
        """Return the flattened representation used by ``jnp.asarray``."""
        return jnp.concatenate(self._components, axis=-1)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        """Return a flattened representation for JAX tree utilities."""
        return self._components, None

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> PulseParams[ParamScalar]:
        """Reconstruct a PulseParams instance from a flattened representation for JAX tree utilities."""
        del aux_data
        self = cast(PulseParams[ParamScalar], object.__new__(cls))
        self._duration, self._detuning_params, self._phase_params, self._rabi_params = children
        return self

    def __repr__(self) -> str:
        """Return a multi-line string representation of the pulse parameters."""
        string_length = 17

        def fmt(name: str, arr: jax.Array) -> str:
            label = f"  {name:<{string_length}} "
            return label + np.array2string(
                np.asarray(arr),
                separator=", ",
                max_line_width=120,
                prefix=" " * len(label),
            )

        return (
            "PulseParams(\n"
            + fmt("duration =", self.duration)
            + ",\n"
            + fmt("detuning_params =", self.detuning_params)
            + ",\n"
            + fmt("phase_params =", self.phase_params)
            + ",\n"
            + fmt("rabi_params =", self.rabi_params)
            + "\n"
            + ")"
        )
