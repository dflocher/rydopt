from __future__ import annotations

import multiprocessing as mp
import sys
import threading
import time
import warnings
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

from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.types import FixedParamsTuple, ParamsTuple

tqdm.monitor_interval = 0

ParamsType = TypeVar("ParamsType", covariant=True)
InfidelityType = TypeVar("InfidelityType", covariant=True)
HistoryType = TypeVar("HistoryType", covariant=True)


@dataclass
class OptimizationResult(Generic[ParamsType, InfidelityType, HistoryType]):
    r"""Data class that stores the results of a gate pulse optimization.

    Attributes:
        params: Final pulse parameters.
        infidelity: Final cost function evaluation.
        history: Cost function evaluations during the optimization.
        num_steps: Number of optimization steps.
        tol: Target gate infidelity.
        duration_in_sec: Duration of the optimization in seconds.

    """

    params: ParamsType  # type: ignore[misc]
    infidelity: InfidelityType  # type: ignore[misc]
    history: HistoryType  # type: ignore[misc]
    num_steps: int
    tol: float
    duration_in_sec: float


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
                        total=self._num_steps, desc=f"proc{proc_idx:02d}", position=proc_idx, file=sys.stdout, ncols=90
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
                    if bar.n < bar.total:
                        bar.n = bar.total
                    bar.close()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _spec(nested: ParamsTuple | FixedParamsTuple) -> tuple[int, ...]:
    return tuple(np.cumsum([len(p) if isinstance(p, Sized) else 1 for p in nested])[:-1].tolist())


def _ravel(nested: ParamsTuple | FixedParamsTuple) -> np.ndarray:
    first, *rest = nested
    return np.concatenate([(first,), *list(rest)])


def _unravel(flat: np.ndarray, split_indices: tuple[int, ...]) -> ParamsTuple | FixedParamsTuple:
    parts = np.split(flat, split_indices)
    return (parts[0][0], *tuple(parts[1:]))


def _unravel_jax(flat: jnp.ndarray, split_indices: tuple[int, ...]) -> ParamsTuple | FixedParamsTuple:
    parts = jnp.split(flat, split_indices)
    return (parts[0][0], *tuple(parts[1:]))  # type: ignore[return-value]


def _make_infidelity(
    gate: Gate,
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
        print("> infidelity <= numerical precision")
    else:
        print(f"> infidelity = {infidelity:.6e}")
    print(f"> parameters = ({', '.join(str(p) for p in params)})")
    print(f"> duration = {params[0]}")


def _print_summary(method_name: str, duration: float, tol: float, num_converged: int):
    print(f"\n=== Optimization finished using {method_name} ===\n")
    print(f"Duration: {duration:.3f} seconds")
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
):
    opt_state0 = optimizer.init(params_trainable)

    def body(carry, step):
        _, _, _, _, prev_converged_initializations = carry

        # Do an gradient descent step if the optimization was not yet done. Note that 'params' and
        # not 'new_params' contains the parameters that correspond to the 'infidelity'.
        def do_step(carry):
            _, params, _, opt_state, _ = carry

            infidelity, grads = infidelity_and_grad(params)
            converged_initializations = jnp.sum(infidelity <= tol)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return (
                params,
                new_params,
                infidelity,
                opt_state,
                converged_initializations,
            )

        was_not_done = prev_converged_initializations < min_converged_initializations
        carry = jax.lax.cond(was_not_done, do_step, lambda carry: carry, operand=carry)

        _, _, infidelity, _, converged_initializations = carry

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

        return carry, infidelity

    (final_params, _, final_infidelity, _, _), infidelity_history = jax.lax.scan(
        body,
        (params_trainable, params_trainable, jnp.zeros_like(tol), opt_state0, 0),
        jnp.arange(num_steps),
    )

    return (final_params, final_infidelity, infidelity_history)


# -----------------------------------------------------------------------------
# Internal Adam optimization helper
# -----------------------------------------------------------------------------


def _adam_optimize(
    gate: Gate,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device_ctx = nullcontext() if device_idx is None else jax.default_device(jax.devices()[device_idx])

    progress_hook = _ProgressBar.make_progress_hook(progress_queue)

    with device_ctx:
        params_trainable = jax.device_put(params_trainable)
        optimizer = optax.adam(learning_rate)
        infidelity = _make_infidelity(
            gate,
            pulse,
            params_full,
            params_trainable_indices,
            params_split_indices,
            tol,
        )

        if params_trainable.ndim == 1:
            infidelity_and_grad = jax.value_and_grad(infidelity)
            tol_arg: float | jnp.ndarray = tol
        else:
            infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))
            tol_arg = jnp.full((params_trainable.shape[0],), tol)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Complex dtype support in Diffrax.*",
                category=UserWarning,
                module=r"^equinox\._jit$",
            )

            final_params, final_infidelities, infidelity_history = _adam_scan(
                infidelity_and_grad=infidelity_and_grad,
                optimizer=optimizer,
                params_trainable=params_trainable,
                num_steps=num_steps,
                min_converged_initializations=min_converged_initializations,
                process_idx=process_idx,
                tol=tol_arg,
                progress_hook=progress_hook,
            )

        return np.array(final_params), np.array(final_infidelities), np.array(infidelity_history)


# -----------------------------------------------------------------------------
# Public optimization functions
# -----------------------------------------------------------------------------


@overload
def optimize(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
    *,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    return_history: Literal[True],
    verbose: bool = ...,
) -> OptimizationResult[ParamsTuple, float, np.ndarray]: ...


@overload
def optimize(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
    *,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    return_history: Literal[False] = False,
    verbose: bool = ...,
) -> OptimizationResult[ParamsTuple, float, None]: ...


def optimize(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = None,
    *,
    num_steps: int = 1000,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
    return_history: bool = False,
    verbose: bool = False,
) -> OptimizationResult[ParamsTuple, float, np.ndarray | None]:
    r"""Function that optimizes an initial parameter guess in order to realize the desired gate.

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

    print("\nStarted optimization using 1 process\n")

    t0 = time.perf_counter()
    with _ProgressBar(
        num_processes=1, num_steps=num_steps, min_converged_initializations=1, enable=not verbose
    ) as progress_queue:
        final_params_trainable, final_infidelity, complete_history = _adam_optimize(
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
        )
        history = complete_history if return_history else None
    duration = time.perf_counter() - t0

    final_full = params_full.copy()
    final_full[trainable_indices] = final_params_trainable

    final_params = _unravel(final_full, split_indices)
    num_converged = 1 if final_infidelity <= tol else 0

    # --- Logging ---

    _print_summary("Adam", duration, tol, num_converged)
    _print_gate("Optimized gate:", final_params, float(final_infidelity), tol)

    return OptimizationResult(
        params=final_params,
        infidelity=float(final_infidelity),
        history=history,
        num_steps=num_steps,
        tol=tol,
        duration_in_sec=duration,
    )


@overload
def multi_start_optimize(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
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
) -> OptimizationResult[list[ParamsTuple], np.ndarray, np.ndarray]: ...


@overload
def multi_start_optimize(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
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
) -> OptimizationResult[list[ParamsTuple], np.ndarray, None]: ...


@overload
def multi_start_optimize(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
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
) -> OptimizationResult[ParamsTuple, float, np.ndarray]: ...


@overload
def multi_start_optimize(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
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
) -> OptimizationResult[ParamsTuple, float, None]: ...


def multi_start_optimize(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = None,
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
) -> OptimizationResult[ParamsTuple | list[ParamsTuple], float | np.ndarray, np.ndarray | None]:
    r"""Function that optimizes multiple random initial parameter guesses in order to realize the desired gate.

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
        if not np.allclose(flat_min[trainable_mask], flat_max[trainable_mask]):
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

    print(f"\nStarted optimization using {num_processes} {'processes' if num_processes > 1 else 'process'}\n")

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
            final_params_trainable, final_infidelities, complete_history = _adam_optimize(
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
            )
            history = complete_history if return_history else None

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
                    )
                    for device_idx, p in enumerate(chunks)
                ],
            )

            # Concatenate results from all processes
            final_params_trainable_list, final_infidelities_list, histories_list = zip(*results)
            final_params_trainable = np.concatenate(final_params_trainable_list, axis=0)
            final_infidelities = np.concatenate(final_infidelities_list, axis=0)
            history = np.concatenate(histories_list, axis=1) if return_history else None

    duration = time.perf_counter() - t0

    final_full = np.tile(params_full, (final_params_trainable.shape[0], 1))
    final_full[:, trainable_indices] = final_params_trainable

    converged = np.where(final_infidelities <= tol)[0]
    num_converged = len(converged)
    if num_converged == 0:
        converged = np.array([np.argmin(final_infidelities)])
    durations_converged = final_full[converged][:, 0]

    # --- Logging ---

    _print_summary("multi-start Adam", duration, tol, num_converged)

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
        history_out = None
        history_out = history[:, sorter] if history is not None else None
        return OptimizationResult(
            params=[_unravel(p, split_indices) for p in final_full[sorter]],
            infidelity=final_infidelities[sorter],
            history=history_out,
            num_steps=num_steps,
            tol=tol,
            duration_in_sec=duration,
        )

    history_out = history[:, fastest_idx] if history is not None else None
    return OptimizationResult(
        params=fastest_params,
        infidelity=final_infidelities[fastest_idx],
        history=history_out,
        num_steps=num_steps,
        tol=tol,
        duration_in_sec=duration,
    )
