from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation import evolve
from functools import partial
import jax.numpy as jnp
import jax
import time
import optax
import numpy as np
from typing import TypeAlias
from jax.flatten_util import ravel_pytree
import multiprocessing as mp
from psutil import cpu_count
import warnings
from dataclasses import dataclass
from collections.abc import Callable


FloatParams: TypeAlias = float | tuple[float, ...]
BoolParams: TypeAlias = bool | tuple[bool, ...]


@dataclass
class _EvolveSetup:
    gate: Gate
    pulse: PulseAnsatz
    tol: float
    unravel: Callable[[jnp.ndarray], object]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _infidelity(params, setup: _EvolveSetup):
    final_states = evolve(setup.gate, setup.pulse, setup.unravel(params), setup.tol)
    return 1 - setup.gate.process_fidelity(final_states)


def _make_grads_mask(fixed_initial_params):
    if fixed_initial_params is None:
        return None
    flat_fixed, _ = ravel_pytree(fixed_initial_params)
    return 1 - np.array(flat_fixed, dtype=float)


def _print_gate(title: str, duration: float, params, infidelity: float):
    print(f"\n{title}")
    if float(infidelity) < 0:
        print("> infidelity <= numerical precision")
    else:
        print(f"> infidelity = {infidelity:.6e}")
    print(f"> parameters = ({', '.join(str(p) for p in params)})")
    print(f"> duration = {duration}")


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
        "use_grad_mask",
        "num_steps",
        "min_converged_initializations",
    ],
    donate_argnames=["initial_params"],
)
def _adam_scan(
    infidelity_and_grad,
    optimizer: optax.GradientTransformation,
    initial_params,
    use_grad_mask,
    grads_mask,
    num_steps: int,
    min_converged_initializations: int,
    tol,
):
    opt_state0 = optimizer.init(initial_params)

    def body(carry, step):
        _, params, _, opt_state, prev_converged_initializations = carry

        def do_step(_):
            new_infidelity, grads = infidelity_and_grad(params)
            new_converged_initializations = jnp.sum(new_infidelity <= tol)

            if use_grad_mask:
                grads = grads * grads_mask

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return (
                params,
                new_params,
                new_infidelity,
                new_opt_state,
                new_converged_initializations,
            )

        def do_no_step(_):
            return carry

        params, new_params, infidelity, opt_state, converged_initializations = (
            jax.lax.cond(
                prev_converged_initializations >= min_converged_initializations,
                do_no_step,
                do_step,
                operand=None,
            )
        )

        is_periodic = (step % 20 == 0) | (step == num_steps - 1)
        log_step = (prev_converged_initializations < min_converged_initializations) & (
            is_periodic | (converged_initializations >= min_converged_initializations)
        )

        jax.lax.cond(
            jnp.any(log_step),
            lambda _: jax.debug.print(
                "Step {step:06d}: min infidelity ={infidelity:13.6e}, converged = {converged}",
                step=step,
                infidelity=jnp.min(infidelity),
                converged=converged_initializations,
            ),
            lambda _: None,
            operand=None,
        )

        return (
            params,
            new_params,
            infidelity,
            opt_state,
            converged_initializations,
        ), infidelity

    (final_params, _, final_infidelity, _, _), infidelity_history = jax.lax.scan(
        body,
        (initial_params, initial_params, jnp.zeros_like(tol), opt_state0, 0),
        jnp.arange(num_steps),
    )

    return final_params, final_infidelity


# -----------------------------------------------------------------------------
# Internal Adam optimization helper
# -----------------------------------------------------------------------------


def _adam_optimize(
    flat_params: np.ndarray,
    grads_mask: np.ndarray | None,
    num_steps: int,
    min_converged_initializations: int,
    learning_rate: float,
    setup: _EvolveSetup,
) -> tuple[np.ndarray, np.ndarray]:
    optimizer = optax.adam(learning_rate)
    infidelity = partial(_infidelity, setup=setup)

    if flat_params.ndim == 1:
        infidelity_and_grad = jax.value_and_grad(infidelity)
        tol = setup.tol
    else:
        infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))
        tol = jnp.full((flat_params.shape[0],), setup.tol)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Complex dtype support in Diffrax.*",
            category=UserWarning,
            module=r"^equinox\._jit$",
        )

        final_params, final_infidelities = _adam_scan(
            infidelity_and_grad=infidelity_and_grad,
            optimizer=optimizer,
            initial_params=jnp.array(flat_params),
            use_grad_mask=(grads_mask is not None),
            grads_mask=None if grads_mask is None else jnp.array(grads_mask),
            num_steps=num_steps,
            min_converged_initializations=min_converged_initializations,
            tol=tol,
        )

    return np.array(final_params), np.array(final_infidelities)


# -----------------------------------------------------------------------------
# Public optimization functions
# -----------------------------------------------------------------------------


def adam(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: tuple[FloatParams, ...],
    fixed_initial_params: tuple[BoolParams, ...] | None = None,
    num_steps: int = 1000,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
) -> tuple[tuple[FloatParams, ...], float]:
    flat_params, unravel = ravel_pytree(initial_params)
    grads_mask = _make_grads_mask(fixed_initial_params)
    setup = _EvolveSetup(gate=gate, pulse=pulse, tol=tol, unravel=unravel)

    # --- Optimize parameters ---

    print("")

    t0 = time.perf_counter()
    final_params, final_infidelity = _adam_optimize(
        flat_params, grads_mask, num_steps, 1, learning_rate, setup
    )
    jax.block_until_ready(final_params)
    duration = time.perf_counter() - t0

    num_converged = 1 if final_infidelity <= tol else 0
    final_params = jax.tree.map(float, unravel(final_params))

    # --- Logging ---

    _print_summary("Adam", duration, tol, num_converged)
    _print_gate("Best gate:", final_params[0], final_params, final_infidelity)

    return final_params, final_infidelity


def multi_start_adam(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params: tuple[FloatParams, ...],
    max_initial_params: tuple[FloatParams, ...],
    fixed_initial_params: tuple[BoolParams, ...] | None = None,
    num_steps: int = 1000,
    num_initializations: int = 10,
    min_converged_initializations: int = 1,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
    seed: int = 0,
    return_all: bool = False,
    num_workers: int | None = None,
) -> (
    tuple[tuple[FloatParams, ...], float]
    | tuple[list[tuple[FloatParams, ...]], list[float]]
):
    flat_min, unravel = ravel_pytree(min_initial_params)
    flat_max, _ = ravel_pytree(max_initial_params)
    grads_mask = _make_grads_mask(fixed_initial_params)
    setup = _EvolveSetup(gate=gate, pulse=pulse, tol=tol, unravel=unravel)

    if num_workers is None:
        n = cpu_count(logical=False)
        num_workers = min(4, (n // 2) if n is not None else 4)

    # Pad the number of initial parameter samples to be a multiple of the number of workers
    pad = (-num_initializations) % num_workers
    padded_num_initializations = num_initializations + pad
    if pad != 0:
        print(
            f"Padding num_initializations from {num_initializations} to "
            f"{padded_num_initializations} to be a multiple of num_workers={num_workers}."
        )

    # Initial parameter samples
    rng = np.random.default_rng(seed)
    flat_params = flat_min + (flat_max - flat_min) * rng.random(
        size=(padded_num_initializations, flat_min.size)
    )

    # --- Optimize parameters ---

    print("")

    t0 = time.perf_counter()

    if num_workers == 1:
        # Run optimization in main process
        final_params, final_infidelities = _adam_optimize(
            flat_params,
            grads_mask,
            num_steps,
            min_converged_initializations,
            learning_rate,
            setup,
        )
        jax.block_until_ready(final_params)

    else:
        # Run optimization in spawned worker processes
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.starmap(
                _adam_optimize,
                [
                    (
                        pc,
                        grads_mask,
                        num_steps,
                        (min_converged_initializations + num_workers - 1)
                        // num_workers,
                        learning_rate,
                        setup,
                    )
                    for pc in np.array_split(flat_params, num_workers, axis=0)
                ],
            )

        # Concatenate results from all workers
        final_params_list, final_infidelities_list = zip(*results)
        final_params = np.concatenate(final_params_list, axis=0)
        final_infidelities = np.concatenate(final_infidelities_list, axis=0)

    duration = time.perf_counter() - t0

    converged = jnp.where(final_infidelities <= tol)[0]
    num_converged = len(converged)
    if num_converged == 0:
        converged = np.array([np.argmin(final_infidelities)])
    params_converged = final_params[converged]
    infidelities_converged = final_infidelities[converged]
    durations_converged = params_converged[:, 0]

    # --- Logging ---

    _print_summary("multi-start Adam", duration, tol, num_converged)

    fastest_idx = np.argmin(durations_converged)
    fastest_duration = durations_converged[fastest_idx]
    fastest_infidelity = infidelities_converged[fastest_idx]
    fastest_params = jax.tree.map(float, unravel(params_converged[fastest_idx]))

    if num_converged > 1:
        # If multiple parameter sets converged, show slowest and fastest gate
        slowest_idx = np.argmax(durations_converged)
        slowest_duration = durations_converged[slowest_idx]
        slowest_infidelity = infidelities_converged[slowest_idx]
        slowest_params = jax.tree.map(float, unravel(params_converged[slowest_idx]))

        _print_gate(
            "Slowest gate:", slowest_duration, slowest_params, slowest_infidelity
        )
        _print_gate(
            "Fastest gate:", fastest_duration, fastest_params, fastest_infidelity
        )

        idx = rng.integers(0, num_converged, size=(1024, num_converged))
        mins = np.asarray(durations_converged)[idx].min(axis=1)
        err = mins.std()
        print(f"> one-sided bootstrap error on duration: {err:.1g}")
    else:
        # Otherwise, show the gate with the smallest infidelity
        _print_gate("Best gate:", fastest_duration, fastest_params, fastest_infidelity)

    # --- Return value(s) ---

    if return_all:
        sorter = np.argsort(final_infidelities)
        return [
            jax.tree.map(float, unravel(p)) for p in final_params[sorter]
        ], final_infidelities[sorter]

    return fastest_params, infidelities_converged[fastest_idx]
