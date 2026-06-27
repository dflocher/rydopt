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
class PulseFamilyParams(Sequence[jax.Array], Generic[ParamScalar]):
    r"""PulseFamily parameter container.

    Stores the four pulse-family parameter groups

    ``(duration_params, detuning_params, phase_params, rabi_params)``.

    For users, each parameter group is exposed through the corresponding
    property with its original shape preserved. Internally, all parameter
    arrays are stored as one-dimensional flattened arrays to support
    efficient concatenation, JAX transformations, and pytree handling.

    The original array shapes are recorded and used to reconstruct the
    public parameter views when accessed through the properties.
    """

    __slots__ = ("_detuning_params", "_duration_params", "_phase_params", "_rabi_params", "_shapes")

    def __init__(
        self,
        duration: JaxArrayLike | npt.ArrayLike = (),
        detuning_params: JaxArrayLike | npt.ArrayLike = (),
        phase_params: JaxArrayLike | npt.ArrayLike = (),
        rabi_params: JaxArrayLike | npt.ArrayLike = (),
    ) -> None:
        duration_arr = jnp.asarray(duration)
        detuning_arr = jnp.asarray(detuning_params)
        phase_arr = jnp.asarray(phase_params)
        rabi_arr = jnp.asarray(rabi_params)

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
    def duration_params(self) -> jax.Array:
        return jnp.asarray(self._duration_params).reshape(self._shapes[0])

    @property
    def detuning_params(self) -> jax.Array:
        return jnp.asarray(self._detuning_params).reshape(self._shapes[1])

    @property
    def phase_params(self) -> jax.Array:
        return jnp.asarray(self._phase_params).reshape(self._shapes[2])

    @property
    def rabi_params(self) -> jax.Array:
        return jnp.asarray(self._rabi_params).reshape(self._shapes[3])

    @property
    def _components(
        self,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return (
            self._duration_params,
            self._detuning_params,
            self._phase_params,
            self._rabi_params,
        )

    @overload
    def __getitem__(self, index: int) -> jax.Array: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[jax.Array]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> jax.Array | Sequence[jax.Array]:
        """Return one parameter component or a sliced tuple of parameter components."""
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

    # ------------------------------------------------------------------ #
    # JAX pytree protocol
    # ------------------------------------------------------------------ #
    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    ]:
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
        aux_data: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        children: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> PulseFamilyParams[ParamScalar]:
        """Reconstruct an instance from (aux_data, children) for JAX tree utilities.

        Does not re-flatten or reshape: children are already the flattened
        arrays and are assigned directly.
        """
        self = cast(PulseFamilyParams[ParamScalar], object.__new__(cls))
        (
            self._duration_params,
            self._detuning_params,
            self._phase_params,
            self._rabi_params,
        ) = children
        self._shapes = aux_data
        return self

    def __repr__(self) -> str:
        """Return a multi-line string representation of the pulse family parameters."""
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
            "PulseFamilyParams(\n"
            + fmt("duration_params =", self.duration_params)
            + ",\n"
            + fmt("detuning_params =", self.detuning_params)
            + ",\n"
            + fmt("phase_params =", self.phase_params)
            + ",\n"
            + fmt("rabi_params =", self.rabi_params)
            + "\n"
            + ")"
        )
