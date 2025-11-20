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

FloatParams: TypeAlias = float | tuple[float, ...]
BoolParams: TypeAlias = bool | tuple[bool, ...]


def _infidelity(params, unravel, gate: Gate, pulse: PulseAnsatz, tol: float):
    final_states = evolve(gate, pulse, unravel(params), tol)
    return 1 - gate.process_fidelity(final_states)


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
def _run_adam(
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
        _, params, infidelity, opt_state, prev_converged_initializations = carry

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


def _run_adam_chunk(
    flat_params_chunk,
    use_grad_mask,
    grads_mask,
    num_steps,
    min_converged_initializations,
    tol,
    learning_rate,
    gate,
    pulse,
    unravel,
):
    tol_chunk = jnp.full((len(flat_params_chunk),), tol)

    optimizer = optax.adam(learning_rate)
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol, unravel=unravel)
    infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Complex dtype support in Diffrax.*",
            category=UserWarning,
            module=r"^equinox\._jit$",
        )

        final_params_chunk, final_infidelities_chunk = _run_adam(
            infidelity_and_grad=infidelity_and_grad,
            optimizer=optimizer,
            initial_params=flat_params_chunk,
            use_grad_mask=use_grad_mask,
            grads_mask=grads_mask,
            num_steps=num_steps,
            min_converged_initializations=min_converged_initializations,
            tol=tol_chunk,
        )

    return final_params_chunk, final_infidelities_chunk


def adam(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: tuple[FloatParams, ...],
    fixed_initial_params: tuple[BoolParams, ...] | None = None,
    num_steps: int = 1000,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
) -> tuple[FloatParams, ...]:
    # Initial parameters
    flat_params, unravel = ravel_pytree(initial_params)
    flat_fixed, _ = ravel_pytree(fixed_initial_params)
    grads_mask = 1 - flat_fixed.astype(float)

    # Optimizer
    optimizer = optax.adam(learning_rate)

    # Infidelity and its gradient
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol, unravel=unravel)
    infidelity_and_grad = jax.value_and_grad(infidelity)

    # --- Optimize parameters ---

    print("")

    t0 = time.perf_counter()
    final_params, final_infidelity = _run_adam(
        infidelity_and_grad=infidelity_and_grad,
        optimizer=optimizer,
        initial_params=flat_params,
        use_grad_mask=(fixed_initial_params is not None),
        grads_mask=grads_mask,
        num_steps=num_steps,
        min_converged_initializations=1,
        tol=tol,
    )
    jax.block_until_ready(final_params)
    duration = time.perf_counter() - t0

    num_converged = 1 if final_infidelity <= tol else 0
    final_params = jax.tree.map(float, unravel(final_params))

    # --- Logging ---

    print("\n=== Optimization finished using Adam ===\n")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Gates with infidelity below tol={tol:.1e}: {num_converged}")

    # Show converged gate
    print("\nBest gate:")
    print(f"> duration = {final_params[0]}")
    print(f"> parameters = ({', '.join(str(p) for p in final_params)})")
    if float(final_infidelity) < 0:
        print("> infidelity <= numerical precision")
    else:
        print(f"> infidelity = {final_infidelity:.6e}")

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
) -> tuple[FloatParams, ...] | list[tuple[FloatParams, ...]]:
    num_workers = cpu_count(logical=False) // 2  # TODO avoid oversubscription

    # Pad the number of initial parameter samples to be a multiple of the number of workers
    num_initializations += (-num_initializations) % num_workers

    # Initial parameter samples
    flat_min, unravel = ravel_pytree(min_initial_params)
    flat_max, _ = ravel_pytree(max_initial_params)
    flat_fixed, _ = ravel_pytree(fixed_initial_params)
    grads_mask = 1 - flat_fixed.astype(float)

    key = jax.random.PRNGKey(seed)
    flat_params = jax.random.uniform(
        key,
        shape=(num_initializations, flat_min.size),
        minval=flat_min,
        maxval=flat_max,
    )

    # --- Optimize parameters ---

    print("")

    t0 = time.perf_counter()

    if num_workers == 1:
        final_params, final_infidelities = _run_adam_chunk(
            flat_params_chunk=flat_params,
            use_grad_mask=(fixed_initial_params is not None),
            grads_maskk=grads_mask,
            num_steps=num_steps,
            min_converged_initializations=min_converged_initializations,
            tol=tol,
            learning_rate=learning_rate,
            gate=gate,
            pulse=pulse,
            unravel=unravel,
        )
        jax.block_until_ready(final_params)

    else:
        # Split across workers
        param_chunks = jnp.array_split(flat_params, num_workers, axis=0)
        per_worker_min_converged_initializations = (
            min_converged_initializations + num_workers - 1
        ) // num_workers

        # Run in spawned worker processes
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.starmap(
                _run_adam_chunk,
                [
                    (
                        pc,
                        (fixed_initial_params is not None),
                        grads_mask,
                        num_steps,
                        per_worker_min_converged_initializations,
                        tol,
                        learning_rate,
                        gate,
                        pulse,
                        unravel,
                    )
                    for pc in param_chunks
                ],
            )

        # Concatenate results from all workers
        final_params_list, final_infidelities_list = zip(*results)
        final_params = jnp.concatenate(final_params_list, axis=0)
        final_infidelities = jnp.concatenate(final_infidelities_list, axis=0)

    duration = time.perf_counter() - t0

    converged = final_infidelities <= tol
    num_converged = int(jnp.sum(converged))
    if num_converged == 0:
        converged = jnp.array([jnp.argmin(final_infidelities)])
    params_converged = final_params[converged]
    infidelities_converged = final_infidelities[converged]
    durations_converged = params_converged[:, 0]

    # --- Logging ---

    print("\n=== Optimization finished using multi-start Adam ===\n")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Gates with infidelity below tol={tol:.1e}: {num_converged}")

    # Show slowest converged gate
    if num_converged > 1:
        slowest_idx = jnp.argmax(durations_converged)
        slowest_duration = durations_converged[slowest_idx]
        slowest_infidelity = final_infidelities[converged][slowest_idx]
        slowest_params = jax.tree.map(float, unravel(params_converged[slowest_idx]))

        print("\nSlowest gate:")
        print(f"> duration = {slowest_duration}")

        print(f"> parameters = ({', '.join(str(p) for p in slowest_params)})")
        if float(slowest_infidelity) < 0:
            print("> infidelity <= numerical precision")
        else:
            print(f"> infidelity = {slowest_infidelity:.6e}")

    # Show fastest converged gate
    fastest_idx = jnp.argmin(durations_converged)
    fastest_duration = durations_converged[fastest_idx]
    fastest_infidelity = final_infidelities[converged][fastest_idx]
    fastest_params = jax.tree.map(float, unravel(params_converged[fastest_idx]))

    if num_converged > 1:
        print("\nFastest gate:")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, num_converged, size=(1024, num_converged))
        mins = np.asarray(durations_converged)[idx].min(axis=1)
        duration_err = mins.std()
        print(
            f"> duration = {fastest_duration} (one-sided bootstrap error: {duration_err:.1g})"
        )
    else:
        print("\nBest gate:")
        print(f"> duration = {fastest_duration}")

    print(f"> parameters = ({', '.join(str(p) for p in fastest_params)})")
    if float(fastest_infidelity) < 0:
        print("> infidelity <= numerical precision")
    else:
        print(f"> infidelity = {fastest_infidelity:.6e}")

    if return_all:
        sorter = jnp.argsort(final_infidelities)
        return [
            jax.tree.map(float, unravel(p)) for p in final_params[sorter]
        ], final_infidelities[sorter]

    return fastest_params, infidelities_converged[fastest_idx]
