from __future__ import annotations

import multiprocessing as mp
import sys
import threading
import time
from collections.abc import Callable, Sized
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from queue import SimpleQueue
from types import TracebackType
from typing import Any, Generic, Literal, Protocol, TypeVar, overload

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from rydopt.protocols import GateSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.types import FixedPulseParams, PulseParams

tqdm.monitor_interval = 0

ParamsType = TypeVar("ParamsType", covariant=True)
ValueType = TypeVar("ValueType", covariant=True)
HistoryType = TypeVar("HistoryType", covariant=True)


@dataclass
class OptimizationResult(Generic[ParamsType, ValueType, HistoryType]):
    r"""Data class that stores the results of a gate pulse optimization.

    Attributes:
        params: Final pulse parameters.
        infidelity: Final cost function evaluation.
        duration: Final duration
        infidelity_history: Cost function evaluations during the optimization.
        duration_history: Durations during the optimization.
        grad_norm_history: Norm of the parameter gradient during the optimization.
        num_steps: Number of optimization steps.
        tol: Target gate infidelity.
        runtime_in_sec: Runtime of the optimization in seconds.

    """

    params: ParamsType  # type: ignore[misc]
    infidelity: ValueType  # type: ignore[misc]
    duration: ValueType  # type: ignore[misc]
    infidelity_history: HistoryType  # type: ignore[misc]
    duration_history: HistoryType  # type: ignore[misc]
    grad_norm_history: HistoryType  # type: ignore[misc]
    num_steps: int
    tol: float
    runtime_in_sec: float


# -----------------------------------------------------------------------------
# Progress bar
# -----------------------------------------------------------------------------


class _ProgressQueue(Protocol):
    def put(self, item: Any) -> None: ...
    def get(self) -> Any: ...


class _ProgressBar:
    def __init__(
        self,
        num_processes: int,
        num_steps: int,
        min_converged_initializations: int,
        queue: _ProgressQueue | None = None,
        enable: bool = True,
    ) -> None:
        self._num_processes = num_processes
        self._num_steps = num_steps
        self._min_converged_initializations = min_converged_initializations
        self._external_queue = queue
        self._queue: _ProgressQueue = queue or SimpleQueue()
        self._listener: threading.Thread | None = None
        self._enable = enable

    def __enter__(self) -> _ProgressQueue | None:
        if not self._enable:
            return None
        self._listener = threading.Thread(
            target=self._progress_listener,
            daemon=True,
        )
        self._listener.start()
        return self._queue

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._enable:
            return
        for proc_idx in range(self._num_processes):
            self._queue.put(("done", proc_idx, 0, 0, 0))
        if self._listener is not None:
            self._listener.join()

    @staticmethod
    def make_progress_hook(
        queue: _ProgressQueue | None,
    ) -> Callable[[tuple[int, int, float, int]], None] | None:
        if queue is None:
            return None

        def progress_hook(args: tuple[int, int, float, int]) -> None:
            process_idx, step, infidelity, converged = args
            queue.put(
                (
                    "update",
                    int(process_idx),
                    int(step),
                    float(infidelity),
                    int(converged),
                )
            )

        return progress_hook

    def _progress_listener(self) -> None:
        bars: dict[int, tqdm] = {}
        finished: set[int] = set()

        while len(finished) < self._num_processes:
            kind, proc_idx, step, min_inf, converged = self._queue.get()

            if kind == "update":
                bar = bars.get(proc_idx)
                if bar is None:
                    bar = tqdm(
                        total=self._num_steps,
                        desc=f"proc{proc_idx:02d}",
                        position=proc_idx,
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )
                    bars[proc_idx] = bar

                bar.n = step + 1
                bar.set_postfix(
                    {
                        "infidelity": f"{min_inf:.2e}",
                        "converged": f"{converged}/{self._min_converged_initializations}",
                    },
                    refresh=False,
                )
                bar.refresh()

            elif kind == "done":
                finished.add(proc_idx)
                bar = bars.pop(proc_idx, None)
                if bar is not None:
                    if bar.n < self._num_steps:
                        bar.n = self._num_steps
                    bar.close()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _spec(nested: PulseParams | FixedPulseParams) -> tuple[int, ...]:
    return tuple(np.cumsum([len(p) if isinstance(p, Sized) else 1 for p in nested])[:-1].tolist())


def _ravel(nested: PulseParams | FixedPulseParams) -> np.ndarray:
    first, *rest = nested
    return np.concatenate([(first,), *list(rest)])


def _unravel(flat: np.ndarray, split_indices: tuple[int, ...]) -> PulseParams | FixedPulseParams:
    parts = np.split(flat, split_indices)
    return (parts[0][0], *tuple(parts[1:]))  # type: ignore[return-value]


def _unravel_jax(flat: jnp.ndarray, split_indices: tuple[int, ...]) -> PulseParams | FixedPulseParams:
    parts = jnp.split(flat, split_indices)
    return (parts[0][0], *tuple(parts[1:]))  # type: ignore[return-value]


def _make_infidelity(
    gate: GateSystem,
    pulse: PulseAnsatz,
    params_full: np.ndarray,
    params_trainable_indices: np.ndarray,
    params_split_indices: tuple,
    tol: float,
):
    full = jnp.asarray(params_full)
    trainable_indices = jnp.asarray(params_trainable_indices)

    def infidelity(params_trainable):
        params = full.at[trainable_indices].set(params_trainable)
        params_tuple = _unravel_jax(params, params_split_indices)
        return jnp.abs(1 - process_fidelity(gate, pulse, params_tuple, tol))

    return infidelity


def _print_gate(title: str, params, infidelity: float, tol: float):
    print(f"\n{title}")
    if abs(float(infidelity)) < tol:
        print("> infidelity <= tol")
    else:
        print(f"> infidelity = {infidelity:.6e}")
    print(f"> parameters = ({', '.join(str(p) for p in params)})")
    print(f"> duration = {params[0]}")


def _print_summary(method_name: str, runtime: float, tol: float, num_converged: int):
    print(f"\n=== Optimization finished using {method_name} ===\n")
    print(f"Runtime: {runtime:.3f} seconds")
    print(f"Gates with infidelity below tol={tol:.1e}: {num_converged}")


# -----------------------------------------------------------------------------
# Internal jax.jit-ed Adam optimization scan loop
# -----------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=[
        "infidelity_and_grad",
        "optimizer",
        "num_steps",
        "min_converged_initializations",
        "progress_hook",
        "return_history",
    ],
    donate_argnames=["params_trainable"],
)
def _adam_scan(
    infidelity_and_grad,
    optimizer: optax.GradientTransformation,
    params_trainable,
    num_steps: int,
    min_converged_initializations: int,
    process_idx: int,
    tol: float | jnp.ndarray,
    progress_hook,
    return_history: bool,
):
    opt_state0 = optimizer.init(params_trainable)

    def body(carry, step):
        _, _, _, _, prev_converged_initializations, _ = carry

        # Do an gradient descent step if the optimization was not yet done. Note that 'params' and
        # not 'new_params' contains the parameters that correspond to the 'infidelity'.
        def do_step(carry):
            _, params, _, opt_state, _, _ = carry

            infidelity, grads = infidelity_and_grad(params)
            converged_initializations = jnp.sum(infidelity <= tol)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            grad_norm = jnp.linalg.norm(grads, axis=-1) if return_history else jnp.zeros_like(tol)

            return (
                params,
                new_params,
                infidelity,
                opt_state,
                converged_initializations,
                grad_norm,
            )

        was_not_done = prev_converged_initializations < min_converged_initializations
        carry = jax.lax.cond(was_not_done, do_step, lambda carry: carry, operand=carry)

        params, _, infidelity, _, converged_initializations, grad_norm = carry

        # Log intermediate results at distinct steps
        is_done_now = converged_initializations >= min_converged_initializations
        is_distinct = (step % 20 == 0) | (step == num_steps - 1)
        should_log = was_not_done & (is_done_now | is_distinct)

        if progress_hook is not None:
            jax.lax.cond(
                should_log,
                lambda args: jax.debug.callback(progress_hook, args),
                lambda _: None,
                operand=(process_idx, step, jnp.min(infidelity), converged_initializations),
            )
        else:
            jax.lax.cond(
                should_log,
                lambda args: jax.debug.print(
                    "Step {step} [proc{process_idx}]: infidelity = {min_infidelity}, "
                    "converged = {converged} / {min_converged_initializations}",
                    step=args[0],
                    process_idx=args[1],
                    min_infidelity=args[2],
                    converged=args[3],
                    min_converged_initializations=args[4],
                ),
                lambda _: None,
                operand=(
                    step,
                    process_idx,
                    jnp.min(infidelity),
                    converged_initializations,
                    min_converged_initializations,
                ),
            )

        if return_history:
            return carry, (infidelity, params[..., 0], grad_norm)
        return carry, None

    (final_params, _, final_infidelity, _, _, _), history = jax.lax.scan(
        body,
        (params_trainable, params_trainable, jnp.zeros_like(tol), opt_state0, 0, jnp.zeros_like(tol)),
        jnp.arange(num_steps),
    )

    return (final_params, final_infidelity, history)


# -----------------------------------------------------------------------------
# Internal Adam optimization helper
# -----------------------------------------------------------------------------


def _adam_optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    params_full: np.ndarray,
    params_trainable: np.ndarray,
    params_trainable_indices: np.ndarray,
    params_split_indices: tuple,
    num_steps: int,
    min_converged_initializations: int,
    learning_rate: float,
    tol: float,
    process_idx: int,
    device_idx: int | None,
    progress_queue: _ProgressQueue | None,
    return_history: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    device_ctx = nullcontext() if device_idx is None else jax.default_device(jax.devices()[device_idx])

    progress_hook = _ProgressBar.make_progress_hook(progress_queue)

    with device_ctx:
        trainable = jnp.asarray(params_trainable)
        optimizer = optax.adam(learning_rate)
        infidelity = _make_infidelity(
            gate,
            pulse,
            params_full,
            params_trainable_indices,
            params_split_indices,
            tol,
        )

        if trainable.ndim == 1:
            infidelity_and_grad = jax.value_and_grad(infidelity)
            tol_arg: float | jnp.ndarray = tol
        else:
            infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))
            tol_arg = jnp.full((trainable.shape[0],), tol)

        final_params, final_infidelities, history = _adam_scan(
            infidelity_and_grad=infidelity_and_grad,
            optimizer=optimizer,
            params_trainable=trainable,
            num_steps=num_steps,
            min_converged_initializations=min_converged_initializations,
            process_idx=process_idx,
            tol=tol_arg,
            progress_hook=progress_hook,
            return_history=return_history,
        )

        if return_history:
            infidelity_history = np.array(history[0])
            duration_history = np.array(history[1])
            grad_norm_history = np.array(history[2])
        else:
            infidelity_history = None
            duration_history = None
            grad_norm_history = None

        return (
            np.array(final_params),
            np.array(final_infidelities),
            infidelity_history,
            duration_history,
            grad_norm_history,
        )


# -----------------------------------------------------------------------------
# Public optimization functions
# -----------------------------------------------------------------------------


@overload
def optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    initial_params: PulseParams,
    fixed_initial_params: FixedPulseParams | None = ...,
    *,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    return_history: Literal[True],
    verbose: bool = ...,
) -> OptimizationResult[PulseParams, float, np.ndarray]: ...


@overload
def optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    initial_params: PulseParams,
    fixed_initial_params: FixedPulseParams | None = ...,
    *,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    return_history: Literal[False] = False,
    verbose: bool = ...,
) -> OptimizationResult[PulseParams, float, None]: ...


def optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    initial_params: PulseParams,
    fixed_initial_params: FixedPulseParams | None = None,
    *,
    num_steps: int = 1000,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
    return_history: bool = False,
    verbose: bool = False,
) -> OptimizationResult[PulseParams, float, np.ndarray | None]:
    r"""Function that optimizes an initial parameter guess in order to realize the desired gate.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.TwoQubitGate(
        ...     phi=None,
        ...     theta=np.pi,
        ...     Vnn=float("inf"),
        ...     decay=0,
        ... )
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> initial_params = (7.6, [-0.1], [1.8, -0.6], [])
        >>> result = ro.optimization.optimize(
        ...     gate,
        ...     pulse,
        ...     initial_params,
        ...     num_steps=200,
        ...     tol=1e-7,
        ... )
        Started optimization ...
        >>> optimized_params = result.params

    Args:
        gate: RydOpt Gate object
        pulse: RydOpt PulseAnsatz object
        initial_params: initial pulse parameters
        fixed_initial_params: which parameters shall not be optimized
        num_steps: number of optimization steps
        learning_rate: optimizer learning rate hyperparameter
        tol: target gate infidelity, also sets the ODE solver tolerance
        return_history: whether or not to return the cost history of the optimization
        verbose: whether detail information is printed or only a progress bar is shown

    Returns:
        OptimizationResult object containing the final parameters, the final cost, and the optimization history

    """
    split_indices = _spec(initial_params)
    params_full = _ravel(initial_params)

    if fixed_initial_params is None:
        trainable_mask = np.ones_like(params_full, dtype=bool)
    else:
        trainable_mask = ~_ravel(fixed_initial_params).astype(bool)
    trainable_indices = np.nonzero(trainable_mask)[0]

    params_trainable = params_full[trainable_indices]

    # --- Optimize parameters ---

    print("Started optimization using 1 process\n")

    t0 = time.perf_counter()
    with _ProgressBar(
        num_processes=1, num_steps=num_steps, min_converged_initializations=1, enable=not verbose
    ) as progress_queue:
        final_params_trainable, final_infidelity, infidelity_history, duration_history, grad_norm_history = (
            _adam_optimize(
                gate,
                pulse,
                params_full,
                params_trainable,
                trainable_indices,
                split_indices,
                num_steps,
                1,
                learning_rate,
                tol,
                0,
                None,
                progress_queue,
                return_history,
            )
        )
    runtime = time.perf_counter() - t0

    final_full = params_full.copy()
    final_full[trainable_indices] = final_params_trainable

    final_params = _unravel(final_full, split_indices)
    num_converged = 1 if final_infidelity <= tol else 0

    # --- Logging ---

    _print_summary("Adam", runtime, tol, num_converged)
    _print_gate("Optimized gate:", final_params, float(final_infidelity), tol)

    return OptimizationResult(
        params=final_params,
        infidelity=float(final_infidelity),
        duration=final_params[0],
        infidelity_history=infidelity_history,
        duration_history=duration_history,
        grad_norm_history=grad_norm_history,
        num_steps=num_steps,
        tol=tol,
        runtime_in_sec=runtime,
    )


@overload
def multi_start_optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    min_initial_params: PulseParams,
    max_initial_params: PulseParams,
    fixed_initial_params: FixedPulseParams | None = ...,
    *,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    num_initializations: int = ...,
    min_converged_initializations: int | None = ...,
    num_processes: int | None = ...,
    seed: int | None = ...,
    return_history: Literal[True],
    return_all: Literal[True],
    verbose: bool = ...,
) -> OptimizationResult[list[PulseParams], np.ndarray, np.ndarray]: ...


@overload
def multi_start_optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    min_initial_params: PulseParams,
    max_initial_params: PulseParams,
    fixed_initial_params: FixedPulseParams | None = ...,
    *,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    num_initializations: int = ...,
    min_converged_initializations: int | None = ...,
    num_processes: int | None = ...,
    seed: int | None = ...,
    return_history: Literal[False] = False,
    return_all: Literal[True],
    verbose: bool = ...,
) -> OptimizationResult[list[PulseParams], np.ndarray, None]: ...


@overload
def multi_start_optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    min_initial_params: PulseParams,
    max_initial_params: PulseParams,
    fixed_initial_params: FixedPulseParams | None = ...,
    *,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    num_initializations: int = ...,
    min_converged_initializations: int | None = ...,
    num_processes: int | None = ...,
    seed: int | None = ...,
    return_history: Literal[True],
    return_all: Literal[False] = False,
    verbose: bool = ...,
) -> OptimizationResult[PulseParams, float, np.ndarray]: ...


@overload
def multi_start_optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    min_initial_params: PulseParams,
    max_initial_params: PulseParams,
    fixed_initial_params: FixedPulseParams | None = ...,
    *,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    num_initializations: int = ...,
    min_converged_initializations: int | None = ...,
    num_processes: int | None = ...,
    seed: int | None = ...,
    return_history: Literal[False] = False,
    return_all: Literal[False] = False,
    verbose: bool = ...,
) -> OptimizationResult[PulseParams, float, None]: ...


def multi_start_optimize(
    gate: GateSystem,
    pulse: PulseAnsatz,
    min_initial_params: PulseParams,
    max_initial_params: PulseParams,
    fixed_initial_params: FixedPulseParams | None = None,
    *,
    num_steps: int = 1000,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
    num_initializations: int = 10,
    min_converged_initializations: int | None = None,
    num_processes: int | None = None,
    seed: int | None = None,
    return_history: bool = False,
    return_all: bool = False,
    verbose: bool = False,
) -> OptimizationResult[PulseParams | list[PulseParams], float | np.ndarray, np.ndarray | None]:
    r"""Function that optimizes multiple random initial parameter guesses in order to realize the desired gate.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.TwoQubitGate(
        ...     phi=None,
        ...     theta=np.pi,
        ...     Vnn=float("inf"),
        ...     decay=0,
        ... )
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> min_initial_params = (6, [-2], [-2, -2], [])
        >>> max_initial_params = (8, [2], [2, 2], [])
        >>> result = ro.optimization.multi_start_optimize(
        ...     gate,
        ...     pulse,
        ...     min_initial_params,
        ...     max_initial_params,
        ...     num_steps=200,
        ...     tol=1e-7,
        ...     num_initializations=10,
        ...     num_processes=1,
        ... )
        Started optimization ...
        >>> optimized_params = result.params

    Args:
        gate: RydOpt Gate object
        pulse: RydOpt PulseAnsatz object
        min_initial_params: lower bound for the random parameter initialization
        max_initial_params: upper bound for the random parameter initialization
        fixed_initial_params: which parameters shall not be optimized
        num_steps: number of optimization steps
        learning_rate: optimizer learning rate hyperparameter
        tol: target gate infidelity, also sets the ODE solver tolerance
        num_initializations: number of runs in the search for gate pulses
        min_converged_initializations: number of runs that must reach ``tol`` for the optimization to stop
        num_processes: number of parallel processes
        seed: seed for the random number generator
        return_history: whether or not to return the cost history of the optimization
        return_all: whether or not to return all optimization results
        verbose: whether detail information is printed or only a progress bar is shown

    Returns:
        OptimizationResult object containing the final parameters, the final cost, and the optimization history

    """
    split_indices = _spec(min_initial_params)
    flat_min = _ravel(min_initial_params)
    flat_max = _ravel(max_initial_params)
    params_full = flat_min.copy()

    if fixed_initial_params is None:
        trainable_mask = np.ones_like(flat_min, dtype=bool)
    else:
        trainable_mask = ~_ravel(fixed_initial_params).astype(bool)
        if not np.allclose(flat_min[~trainable_mask], flat_max[~trainable_mask]):
            raise ValueError(
                "For fixed parameters, min_initial_params and max_initial_params must have identical values."
            )
    trainable_indices = np.nonzero(trainable_mask)[0]

    use_one_process_per_device = len(jax.devices()) > 1 or jax.devices()[0].platform != "cpu"
    if num_processes is None:
        num_processes = (
            len(jax.devices()) if use_one_process_per_device else max(1, mp.cpu_count() // 2)
        )  # the division by 2 avoids oversubscription
    elif use_one_process_per_device and num_processes > len(jax.devices()):
        raise ValueError(
            "If multiple devices or a GPU device is visible, num_processes must be smaller or equal "
            "to the number of devices."
        )

    # Pad the number of initial parameter samples to be a multiple of the number of processes
    pad = (-num_initializations) % num_processes
    padded_num_initializations = num_initializations + pad
    if pad != 0:
        print(
            f"Padding num_initializations from {num_initializations} to "
            f"{padded_num_initializations} to be a multiple of num_processes={num_processes}."
        )

    if min_converged_initializations is None:
        min_converged_initializations = padded_num_initializations

    # Initial parameter samples
    rng = np.random.default_rng(seed)
    params_trainable = flat_min[trainable_indices] + (
        flat_max[trainable_indices] - flat_min[trainable_indices]
    ) * rng.random(size=(padded_num_initializations, trainable_indices.size))

    # --- Optimize parameters ---

    print(f"Started optimization using {num_processes} {'processes' if num_processes > 1 else 'process'}\n")

    t0 = time.perf_counter()

    min_converged_initializations_local = (min_converged_initializations + num_processes - 1) // num_processes

    if num_processes == 1:
        # Run optimization in main process
        with _ProgressBar(
            num_processes=num_processes,
            num_steps=num_steps,
            min_converged_initializations=min_converged_initializations_local,
            enable=not verbose,
        ) as progress_queue:
            final_params_trainable, final_infidelities, infidelity_history, duration_history, grad_norm_history = (
                _adam_optimize(
                    gate,
                    pulse,
                    params_full,
                    params_trainable,
                    trainable_indices,
                    split_indices,
                    num_steps,
                    min_converged_initializations_local,
                    learning_rate,
                    tol,
                    0,
                    None,
                    progress_queue,
                    return_history,
                )
            )

    else:
        # Run optimization in spawned processes
        chunks = np.array_split(params_trainable, num_processes, axis=0)

        ctx = mp.get_context("spawn")
        with (
            ctx.Manager() as manager,
            _ProgressBar(
                num_processes=num_processes,
                num_steps=num_steps,
                min_converged_initializations=min_converged_initializations_local,
                queue=manager.Queue(),
                enable=not verbose,
            ) as progress_queue,
            ctx.Pool(processes=num_processes) as pool,
        ):
            results = pool.starmap(
                _adam_optimize,
                [
                    (
                        gate,
                        pulse,
                        params_full,
                        p,
                        trainable_indices,
                        split_indices,
                        num_steps,
                        min_converged_initializations_local,
                        learning_rate,
                        tol,
                        device_idx,
                        device_idx if use_one_process_per_device else None,
                        progress_queue,
                        return_history,
                    )
                    for device_idx, p in enumerate(chunks)
                ],
            )

            # Concatenate results from all processes
            (
                final_params_trainable_list,
                final_infidelities_list,
                infidelity_history_list,
                duration_history_list,
                grad_norm_history_list,
            ) = zip(*results)
            final_params_trainable = np.concatenate(final_params_trainable_list, axis=0)
            final_infidelities = np.concatenate(final_infidelities_list, axis=0)

            if return_history:
                infidelity_history = np.concatenate(infidelity_history_list, axis=1)
                duration_history = np.concatenate(duration_history_list, axis=1)
                grad_norm_history = np.concatenate(grad_norm_history_list, axis=1)
            else:
                infidelity_history = None
                duration_history = None
                grad_norm_history = None

    runtime = time.perf_counter() - t0

    final_full = np.tile(params_full, (final_params_trainable.shape[0], 1))
    final_full[:, trainable_indices] = final_params_trainable

    converged = np.where(final_infidelities <= tol)[0]
    num_converged = len(converged)
    if num_converged == 0:
        converged = np.array([np.argmin(final_infidelities)])
    durations_converged = final_full[converged][:, 0]

    # --- Logging ---

    _print_summary("multi-start Adam", runtime, tol, num_converged)

    fastest_idx = converged[np.argmin(durations_converged)]
    fastest_infidelity = final_infidelities[fastest_idx]
    fastest_params = _unravel(final_full[fastest_idx], split_indices)

    if num_converged > 1:
        # If multiple parameter sets converged, show slowest and fastest gate
        slowest_idx = converged[np.argmax(durations_converged)]
        slowest_infidelity = final_infidelities[slowest_idx]
        slowest_params = _unravel(final_full[slowest_idx], split_indices)

        _print_gate("Slowest gate:", slowest_params, slowest_infidelity, tol)
        _print_gate("Fastest gate:", fastest_params, fastest_infidelity, tol)

        idx = rng.integers(0, num_converged, size=(1024, num_converged))
        mins = np.asarray(durations_converged)[idx].min(axis=1)
        err = mins.std()
        print(f"> one-sided bootstrap error on duration: {err:.1g}")
    else:
        # Otherwise, show the gate with the smallest infidelity
        _print_gate("Best gate:", fastest_params, fastest_infidelity, tol)

    # --- Return value(s) ---

    if return_all:
        sorter = np.argsort(final_infidelities)
        final_full_sorted = final_full[sorter]
        infidelity_history_out = infidelity_history[:, sorter] if infidelity_history is not None else None
        duration_history_out = duration_history[:, sorter] if duration_history is not None else None
        grad_norm_history_out = grad_norm_history[:, sorter] if grad_norm_history is not None else None
        return OptimizationResult(
            params=[_unravel(p, split_indices) for p in final_full_sorted],
            infidelity=final_infidelities[sorter],
            duration=final_full_sorted[:, 0],
            infidelity_history=infidelity_history_out,
            duration_history=duration_history_out,
            grad_norm_history=grad_norm_history_out,
            num_steps=num_steps,
            tol=tol,
            runtime_in_sec=runtime,
        )

    infidelity_history_out = infidelity_history[:, fastest_idx] if infidelity_history is not None else None
    duration_history_out = duration_history[:, fastest_idx] if duration_history is not None else None
    grad_norm_history_out = grad_norm_history[:, fastest_idx] if grad_norm_history is not None else None
    return OptimizationResult(
        params=fastest_params,
        infidelity=final_infidelities[fastest_idx],
        duration=fastest_params[0],
        infidelity_history=infidelity_history_out,
        duration_history=duration_history_out,
        grad_norm_history=grad_norm_history_out,
        num_steps=num_steps,
        tol=tol,
        runtime_in_sec=runtime,
    )
