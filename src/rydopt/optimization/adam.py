from __future__ import annotations

from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from functools import partial
import jax.numpy as jnp
import jax
import time
import optax
import numpy as np
import multiprocessing as mp
import warnings
from dataclasses import dataclass
from contextlib import nullcontext
from rydopt.types import ParamsTuple, FixedParamsTuple
from typing import Generic, TypeVar, overload, Literal

ParamsT = TypeVar("ParamsT", covariant=True)
InfidelityT = TypeVar("InfidelityT", covariant=True)
HistoryT = TypeVar("HistoryT", covariant=True)


# TODO: Shall we add some optimization parameters such as tol and num_steps to the data class?
#  One could then set up a 'plot_history' function in the ro.characterization module, which just takes an OptimizationResult object as input
@dataclass
class OptimizationResult(Generic[ParamsT, InfidelityT, HistoryT]):
    r"""Data class that stores the results of a gate pulse optimization.

    Attributes:
        params: Final pulse parameters.
        infidelity: Final cost function evaluation.
        history: Cost function evaluations during the optimization.

    """

    params: ParamsT  # type: ignore[misc]
    infidelity: InfidelityT  # type: ignore[misc]
    history: HistoryT  # type: ignore[misc]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _spec(nested):
    first, *rest = nested
    return tuple(np.cumsum([1] + [len(p) for p in rest])[:-1].tolist())


def _ravel(nested):
    first, *rest = nested
    return np.concatenate([(first,)] + [p for p in rest])


def _unravel(flat, split_indices):
    parts = np.split(flat, split_indices)
    return (parts[0][0],) + tuple(parts[1:])


def _unravel_jax(flat, split_indices):
    parts = jnp.split(flat, split_indices)
    return (parts[0][0],) + tuple(parts[1:])


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
        return 1 - process_fidelity(
            gate, pulse, params_tuple, tol
        )  # TODO: My IDE's typechecker complains about params_tuple

    return infidelity


def _print_gate(title: str, params, infidelity: float):
    print(f"\n{title}")
    # TODO: Already if infidelity \approx tol, the number printed here might not be quite correct (see example gate_optimization_single.py)
    if float(infidelity) < 0:
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
    tol,
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
        jax.lax.cond(
            was_not_done & (is_done_now | is_distinct),
            lambda args: jax.debug.print(
                "Step {step:06d} [process{process_idx:02d}]: min infidelity ={min_infidelity:13.6e}, converged = {converged} / {min_converged_initializations}",
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
    device_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device_ctx = (
        nullcontext()
        if device_idx is None
        else jax.default_device(jax.devices()[device_idx])
    )

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
            tol_arg = tol
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
            )

        return (
            np.array(final_params),
            np.array(final_infidelities),
            np.array(infidelity_history),
        )


# -----------------------------------------------------------------------------
# Public optimization functions
# -----------------------------------------------------------------------------


@overload
def adam(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    *,
    return_history: Literal[True],
) -> OptimizationResult[ParamsTuple, float, np.ndarray]: ...


@overload
def adam(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
    num_steps: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    *,
    return_history: Literal[False] = False,
) -> OptimizationResult[ParamsTuple, float, None]: ...


# TODO: I'd rename the function to something like 'optimize'.
#  One could just as well use another optimizer, such as adagrad or nadam.
#  This would probably just change a single line of code: optax.adam -> optax.adagrad
#  Being able to specify a concrete optimizer could be a future upgrade of the package
def adam(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = None,
    num_steps: int = 1000,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
    *,
    return_history: bool = False,
) -> OptimizationResult[ParamsTuple, float, np.ndarray | None]:
    r"""Function that optimizes an initial parameter guess in order to realize the desired gate.

    Args:
        gate: rydopt Gate object
        pulse: rydopt PulseAnsatz object
        initial_params: initial pulse parameters
        fixed_initial_params: which parameters shall not be optimized
        num_steps: number of optimization steps
        learning_rate: optimizer learning rate hyperparameter
        tol: precision of the ODE solver
        return_history: whether or not to return the cost history of the optimization

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

    print("\nStarted optimization using 1 process")

    t0 = time.perf_counter()
    final_params_trainable, final_infidelity, history = _adam_optimize(
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
    )
    duration = time.perf_counter() - t0

    final_full = params_full.copy()
    final_full[trainable_indices] = final_params_trainable

    final_params = _unravel(final_full, split_indices)
    num_converged = 1 if final_infidelity <= tol else 0

    # --- Logging ---

    _print_summary("Adam", duration, tol, num_converged)
    _print_gate("Optimized gate:", final_params, float(final_infidelity))

    history_out = history if return_history else None
    return OptimizationResult(
        params=final_params, infidelity=float(final_infidelity), history=history_out
    )


@overload
def multi_start_adam(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
    num_steps: int = ...,
    num_initializations: int = ...,
    min_converged_initializations: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    num_processes: int | None = ...,
    seed: int = ...,
    *,
    return_history: Literal[True],
    return_all: Literal[True],
) -> OptimizationResult[list[ParamsTuple], np.ndarray, np.ndarray]: ...


@overload
def multi_start_adam(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
    num_steps: int = ...,
    num_initializations: int = ...,
    min_converged_initializations: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    num_processes: int | None = ...,
    seed: int = ...,
    *,
    return_history: Literal[False] = False,
    return_all: Literal[True],
) -> OptimizationResult[list[ParamsTuple], np.ndarray, None]: ...


@overload
def multi_start_adam(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
    num_steps: int = ...,
    num_initializations: int = ...,
    min_converged_initializations: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    num_processes: int | None = ...,
    seed: int = ...,
    *,
    return_history: Literal[True],
    return_all: Literal[False] = False,
) -> OptimizationResult[ParamsTuple, float, np.ndarray]: ...


@overload
def multi_start_adam(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = ...,
    num_steps: int = ...,
    num_initializations: int = ...,
    min_converged_initializations: int = ...,
    learning_rate: float = ...,
    tol: float = ...,
    num_processes: int | None = ...,
    seed: int = ...,
    *,
    return_history: Literal[False] = False,
    return_all: Literal[False] = False,
) -> OptimizationResult[ParamsTuple, float, None]: ...


# TODO: rename function (see my comment above)
def multi_start_adam(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: ParamsTuple,
    max_initial_params: ParamsTuple,
    fixed_initial_params: FixedParamsTuple | None = None,
    num_steps: int = 1000,  # TODO: Do you mind if change the order to ..., num_steps, learning_rate, tol, num_initializations, ... to match the adam function?
    num_initializations: int = 10,
    min_converged_initializations: int = 1,  # TODO: I think a better default value would be min_converged_initializations=num_initializations
    learning_rate: float = 0.05,
    tol: float = 1e-7,
    num_processes: int | None = None,
    seed: int = 0,  # TODO: Can one set the default to a 'random' number such as time.time_ns() ?
    *,
    return_history: bool = False,
    return_all: bool = False,
) -> OptimizationResult[
    ParamsTuple | list[ParamsTuple], float | np.ndarray, np.ndarray | None
]:
    r"""Function that optimizes multiple random initial parameter guesses in order to realize the desired gate.

    Args:
        gate: rydopt Gate object
        pulse: rydopt PulseAnsatz object
        min_initial_params: lower bound for the random parameter initialization
        max_initial_params: upper bound for the random parameter initialization
        fixed_initial_params: which parameters shall not be optimized
        num_steps: number of optimization steps
        num_initializations: number of runs in the search for gate pulses
        min_converged_initializations: this function is terminated if this many runs have reached the specified infidelity
        learning_rate: optimizer learning rate hyperparameter
        tol: desired gate infidelity precision of the ODE solver
        num_processes: number of parallel workers on a multi-core CPU
        seed: seed for random number generator
        return_history: whether or not to return the cost history of the optimization
        return_all: whether or not to return all optimization results

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

    use_one_process_per_device = (
        len(jax.devices()) > 1 or jax.devices()[0].platform != "cpu"
    )
    if num_processes is None:
        num_processes = (
            len(jax.devices())
            if use_one_process_per_device
            else max(1, mp.cpu_count() // 2)
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

    # Initial parameter samples
    rng = np.random.default_rng(seed)
    params_trainable = flat_min[trainable_indices] + (
        flat_max[trainable_indices] - flat_min[trainable_indices]
    ) * rng.random(size=(padded_num_initializations, trainable_indices.size))

    # --- Optimize parameters ---

    print(
        f"\nStarted optimization using {num_processes} {'processes' if num_processes > 1 else 'process'}"
    )

    t0 = time.perf_counter()

    if num_processes == 1:
        # Run optimization in main process
        final_params_trainable, final_infidelities, history = _adam_optimize(
            gate,
            pulse,
            params_full,
            params_trainable,
            trainable_indices,
            split_indices,
            num_steps,
            min_converged_initializations,
            learning_rate,
            tol,
            0,
        )

    else:
        # Run optimization in spawned processes
        chunks = np.array_split(params_trainable, num_processes, axis=0)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_processes) as pool:
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
                        (min_converged_initializations + num_processes - 1)
                        // num_processes,
                        learning_rate,
                        tol,
                        device_idx,
                        device_idx if use_one_process_per_device else None,
                    )
                    for device_idx, p in enumerate(chunks)
                ],
            )

        # Concatenate results from all processes
        final_params_trainable_list, final_infidelities_list, histories_list = zip(
            *results
        )
        final_params_trainable = np.concatenate(final_params_trainable_list, axis=0)
        final_infidelities = np.concatenate(final_infidelities_list, axis=0)

        if return_history:
            history = np.concatenate(histories_list, axis=1)

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

        _print_gate("Slowest gate:", slowest_params, slowest_infidelity)
        _print_gate("Fastest gate:", fastest_params, fastest_infidelity)

        idx = rng.integers(0, num_converged, size=(1024, num_converged))
        mins = np.asarray(durations_converged)[idx].min(axis=1)
        err = mins.std()
        print(f"> one-sided bootstrap error on duration: {err:.1g}")
    else:
        # Otherwise, show the gate with the smallest infidelity
        _print_gate("Best gate:", fastest_params, fastest_infidelity)

    # --- Return value(s) ---

    if return_all:
        sorter = np.argsort(final_infidelities)
        history_out = None
        history_out = (
            history[:, sorter] if return_history else None
        )  # TODO: My IDE says history might be referenced before assignment
        return OptimizationResult(
            params=[_unravel(p, split_indices) for p in final_full[sorter]],
            infidelity=final_infidelities[sorter],
            history=history_out,
        )

    history_out = history[:, fastest_idx] if return_history else None  # TODO: see above
    return OptimizationResult(
        params=fastest_params,
        infidelity=final_infidelities[fastest_idx],
        history=history_out,
    )
