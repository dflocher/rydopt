from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeVar, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

ParamScalar = TypeVar("ParamScalar", float, bool)


@jax.tree_util.register_pytree_node_class
class PulseFamilyParams(Sequence[Any], Generic[ParamScalar]):
    r"""PulseFamily-parameter container.

    Stores the pulse family parameter components and their original shapes:
    ``(duration, detuning_params, phase_params, rabi_params)`` plus ``shapes``.

    Arrays are stored flattened internally; ``shapes`` records the original
    shapes so they can be restored via :meth:`unflatten`.
    """

    __slots__ = ("_detuning_params", "_duration_params", "_phase_params", "_rabi_params", "_shapes")

    def __init__(
        self,
        duration: npt.ArrayLike = (),
        detuning_params: npt.ArrayLike = (),
        phase_params: npt.ArrayLike = (),
        rabi_params: npt.ArrayLike = (),
    ) -> None:
        duration_arr = np.asarray(duration)
        detuning_arr = np.asarray(detuning_params)
        phase_arr = np.asarray(phase_params)
        rabi_arr = np.asarray(rabi_params)

        # Store flattened arrays internally.
        self._duration_params = duration_arr.reshape(-1)
        self._detuning_params = detuning_arr.reshape(-1)
        self._phase_params = phase_arr.reshape(-1)
        self._rabi_params = rabi_arr.reshape(-1)

        # Store original shapes so they can be restored.
        self._shapes = (
            duration_arr.shape,
            detuning_arr.shape,
            phase_arr.shape,
            rabi_arr.shape,
        )

    def __len__(self) -> int:
        """Return the number of parameter components."""
        return 4

    @property
    def _components(
        self,
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
    ]:
        return (
            self._duration_params,
            self._detuning_params,
            self._phase_params,
            self._rabi_params,
        )

    @overload
    def __getitem__(self, index: Literal[0, 1, 2, 3]) -> npt.NDArray[Any]: ...

    @overload
    def __getitem__(self, index: int) -> npt.NDArray[Any]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[npt.NDArray[Any], ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], ...]:
        """Return one parameter component or a sliced tuple of parameter components."""
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

    # ------------------------------------------------------------------ #
    # JAX pytree protocol
    # ------------------------------------------------------------------ #
    def tree_flatten(self) -> tuple[tuple[Any, Any, Any, Any], tuple[Any, ...]]:
        """Return (children, aux_data) for JAX tree utilities.

        Children are the four flattened arrays (these are traced/transformed);
        aux_data holds the static original shapes.
        """
        children = self._components
        aux_data = self._shapes
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[Any, ...],
        children: tuple[Any, Any, Any, Any],
    ) -> PulseFamilyParams[Any]:
        """Reconstruct an instance from (aux_data, children) for JAX tree utilities.

        Does not re-flatten or reshape: children are already the flattened
        arrays and are assigned directly.
        """
        self = object.__new__(cls)
        (
            self._duration_params,
            self._detuning_params,
            self._phase_params,
            self._rabi_params,
        ) = children
        self._shapes = aux_data
        return self

    def __repr__(self) -> str:
        """Return a multi-line string representation of the pulse parameters."""

        def fmt(name: str, flat_arr: npt.NDArray[Any], shape: tuple[int, ...]) -> str:
            prefix = f"  {name}="
            arr = np.asarray(flat_arr).reshape(shape)
            return np.array2string(
                arr,
                separator=", ",
                max_line_width=120,
                prefix=prefix,
            )

        return (
            "PulseFamilyParams(\n"
            f"  duration={fmt('duration', self._duration_params, self._shapes[0])},\n"
            f"  detuning_params={fmt('detuning_params', self._detuning_params, self._shapes[1])},\n"
            f"  phase_params={fmt('phase_params', self._phase_params, self._shapes[2])},\n"
            f"  rabi_params={fmt('rabi_params', self._rabi_params, self._shapes[3])}\n"
            ")"
        )
