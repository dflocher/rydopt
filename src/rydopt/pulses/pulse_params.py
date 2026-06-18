from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeVar, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

ParamScalar = TypeVar("ParamScalar", float, bool)


@jax.tree_util.register_pytree_node_class
class PulseParams(Sequence[Any], Generic[ParamScalar]):
    r"""Pulse-parameter container.

    The container stores pulse parameters components
    ``(duration, detuning_params, phase_params, rabi_params)``.
    """

    __slots__ = ("_detuning_params", "_duration", "_phase_params", "_rabi_params")

    def __init__(
        self,
        duration: ParamScalar,
        detuning_params: npt.ArrayLike = (),
        phase_params: npt.ArrayLike = (),
        rabi_params: npt.ArrayLike = (),
    ) -> None:
        self._duration = np.asarray(duration).reshape(1)
        self._detuning_params = np.asarray(detuning_params).reshape(-1)
        self._phase_params = np.asarray(phase_params).reshape(-1)
        self._rabi_params = np.asarray(rabi_params).reshape(-1)

    def __len__(self) -> int:
        """Return the number of parameter components."""
        return 4

    @property
    def _components(
        self,
    ) -> tuple[
        Any,
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
    ]:
        return (
            self._duration,
            self._detuning_params,
            self._phase_params,
            self._rabi_params,
        )

    @overload
    def __getitem__(self, index: Literal[0]) -> ParamScalar: ...

    @overload
    def __getitem__(self, index: Literal[1, 2, 3]) -> npt.NDArray[Any]: ...

    @overload
    def __getitem__(self, index: int) -> ParamScalar | npt.NDArray[Any]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[npt.NDArray[Any], ...]: ...

    def __getitem__(self, index: int | slice) -> ParamScalar | npt.NDArray[Any] | tuple[npt.NDArray[Any], ...]:
        """Return one parameter component or a sliced tuple of parameter components."""
        if isinstance(index, int) and index == 0:
            return self._duration[0]
        return self._components[index]

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_]:
        """Return the flattened representation used by ``np.asarray``."""
        del dtype
        array = np.concatenate(self._components)
        if copy:
            return array.copy()

        return array

    def __jax_array__(self) -> jax.Array:
        """Return the flattened representation used by ``jnp.asarray``."""
        return jnp.concatenate(self._components, axis=-1)

    def tree_flatten(self) -> tuple[tuple[Any, Any, Any, Any], None]:
        """Return a flattened representation for JAX tree utilities."""
        return self._components, None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, Any, Any, Any]) -> PulseParams[Any]:
        """Reconstruct a PulseParams instance from a flattened representation for JAX tree utilities."""
        del aux_data
        self = cast(PulseParams[Any], object.__new__(cls))
        self._duration, self._detuning_params, self._phase_params, self._rabi_params = children
        return self

    def __repr__(self) -> str:
        """Return a multi-line string representation of the pulse parameters."""

        def fmt(name: str, arr: npt.NDArray[Any]) -> str:
            prefix = f"  {name}="
            return np.array2string(
                arr,
                separator=", ",
                max_line_width=120,
                prefix=prefix,
            )

        return (
            "PulseParams(\n"
            f"  duration={fmt('duration', self._duration)},\n"
            f"  detuning_params={fmt('detuning_params', self._detuning_params)},\n"
            f"  phase_params={fmt('phase_params', self._phase_params)},\n"
            f"  rabi_params={fmt('rabi_params', self._rabi_params)}\n"
            ")"
        )
