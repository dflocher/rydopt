from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation import evolve
from functools import partial
import jax.numpy as jnp
import jax.sharding as jsr
import jax
import time
import optax
import numpy as np
from typing import TypeAlias
from jax.flatten_util import ravel_pytree

FloatParams: TypeAlias = float | tuple[float, ...]
BoolParams: TypeAlias = bool | tuple[bool, ...]


def _infidelity(params, unravel, gate: Gate, pulse: PulseAnsatz, tol: float):
    final_states = evolve(gate, pulse, unravel(params), tol)
    return 1 - gate.process_fidelity(final_states)


"""
@partial(
    jax.jit,
    static_argnames=[
        "infidelity_and_grad",
        "optimizer",
        "num_steps",
        "min_converged_initializations",
    ],
    donate_argnames=["initial_params"],
)
def _adam_runner(
    infidelity_and_grad,
    optimizer: optax.GradientTransformation,
    initial_params,
    num_steps: int,
    min_converged_initializations: int,
    tol,
):
    opt_state0 = optimizer.init(initial_params)

    def body(carry, step):
        params, opt_state, prev_converged_initializations, steps_taken = carry

        infidelity, grads = infidelity_and_grad(params)
        new_converged_initializations = 1
        new_steps_taken = steps_taken + 1

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state, new_converged_initializations, new_steps_taken), infidelity

    (final_params, _, converged_initializations, steps_taken), infidelity_history = (
        jax.lax.scan(
            body,
            (initial_params, opt_state0, 0, 0),
            jnp.arange(num_steps),
        )
    )

    return final_params, infidelity_history, steps_taken, converged_initializations
"""


@partial(
    jax.jit,
    static_argnames=[
        "infidelity_and_grad",
        "optimizer",
        "num_steps",
        "min_converged_initializations",
    ],
    donate_argnames=["initial_params"],
)
def _adam_runner(
    infidelity_and_grad,
    optimizer: optax.GradientTransformation,
    initial_params,
    num_steps: int,
    min_converged_initializations: int,
    tol,
):
    opt_state0 = optimizer.init(initial_params)

    def body(carry, step):
        params, opt_state, prev_converged_initializations, steps_taken = carry

        def do_step(_):
            infidelity, grads = infidelity_and_grad(params)
            new_converged_initializations = jnp.sum(infidelity <= tol)
            new_steps_taken = steps_taken + 1

            def apply_update(_):
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state

            def keep_params(_):
                return params, opt_state

            new_params, new_opt_state = jax.lax.cond(
                new_converged_initializations >= min_converged_initializations,
                keep_params,
                apply_update,
                operand=None,
            )

            return (
                new_params,
                new_opt_state,
                new_converged_initializations,
                new_steps_taken,
            ), infidelity

        def do_no_step(_):
            return carry, jnp.zeros_like(tol)

        (params, opt_state, converged_initializations, steps_taken), infidelity = (
            jax.lax.cond(
                prev_converged_initializations >= min_converged_initializations,
                do_no_step,
                do_step,
                operand=None,
            )
        )

        is_periodic = (step % 10 == 0) | (step == num_steps - 1)
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

        return (params, opt_state, converged_initializations, steps_taken), infidelity

    (final_params, _, converged_initializations, steps_taken), infidelity_history = (
        jax.lax.scan(
            body,
            (initial_params, opt_state0, 0, 0),
            jnp.arange(num_steps),
        )
    )

    return final_params, infidelity_history, steps_taken, converged_initializations


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

    # Optimizer
    if fixed_initial_params is None:
        optimizer = optax.adam(learning_rate)
    else:
        flat_fixed, _ = ravel_pytree(fixed_initial_params)
        optimizer = optax.chain(
            optax.adam(learning_rate),
            # TODO optax.transforms.freeze(flat_fixed), fix it !!!!!!!!
        )

    # Infidelity and its gradient
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol, unravel=unravel)
    infidelity_and_grad = jax.value_and_grad(infidelity)

    # --- Optimize parameters ---

    print("")

    t0 = time.perf_counter()
    final_params, infidelity_history, steps_taken, converged_initializations = (
        _adam_runner(
            infidelity_and_grad=infidelity_and_grad,
            optimizer=optimizer,
            initial_params=flat_params,
            num_steps=num_steps,
            min_converged_initializations=1,
            tol=tol,
        )
    )
    jax.block_until_ready(final_params)
    duration = time.perf_counter() - t0

    if converged_initializations == 0:
        raise RuntimeError("No convergence. Try increasing num_steps or relaxing tol.")

    final_params = jax.tree.map(float, unravel(final_params))

    # --- Logging ---

    print("\n=== Optimization finished using Adam ===\n")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Steps taken: {steps_taken}")
    print(f"Convergences: {converged_initializations}")

    # Show converged gate
    infidelity = infidelity_history[steps_taken - 1]

    print("\nConverged gate:")
    print(f"> duration = {final_params[0]}")
    print(f"> parameters = ({', '.join(str(p) for p in final_params)})")
    if float(infidelity) < 0:
        print("> infidelity <= numerical precision")
    else:
        print(f"> infidelity = {infidelity:.6e}")

    return final_params


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
    return_all_converged: bool = False,
) -> tuple[FloatParams, ...] | list[tuple[FloatParams, ...]]:
    # Initial parameter samples
    key = jax.random.PRNGKey(seed)
    flat_min, unravel = ravel_pytree(min_initial_params)
    flat_max, _ = ravel_pytree(max_initial_params)
    flat_params = jax.random.uniform(
        key,
        shape=(num_initializations, flat_min.size),
        minval=flat_min,
        maxval=flat_max,
    )

    tol_vec = jnp.full((num_initializations,), tol)

    # Optimizer
    if fixed_initial_params is None:
        optimizer = optax.adam(learning_rate)
    else:
        optimizer = optax.chain(
            optax.adam(learning_rate),
            optax.transforms.freeze(ravel_pytree(fixed_initial_params)[0]),
        )

    # Infidelity and its gradient
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol, unravel=unravel)
    infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))

    # --- Optimize parameters ---

    print("")

    devices = jax.devices()
    if len(devices) > 1:
        mesh = jax.make_mesh((len(devices),), ("s",))
        sharding = jsr.NamedSharding(mesh, jsr.PartitionSpec("s"))
        flat_params = jax.device_put(flat_params, sharding)
        tol_vec = jax.device_put(tol_vec, sharding)

    t0 = time.perf_counter()
    final_params, infidelity_history, steps_taken, converged_initializations = (
        _adam_runner(
            infidelity_and_grad=infidelity_and_grad,
            optimizer=optimizer,
            initial_params=flat_params,
            num_steps=num_steps,
            min_converged_initializations=min_converged_initializations,
            tol=tol_vec,
        )
    )
    jax.block_until_ready(final_params)
    duration = time.perf_counter() - t0

    if converged_initializations == 0:
        raise RuntimeError("No convergence. Try increasing num_steps or relaxing tol.")

    final_infidelities = infidelity_history[steps_taken - 1]
    converged = final_infidelities <= tol
    num_converged = int(jnp.sum(converged))
    params_converged = final_params[converged]
    durations_converged = params_converged[:, 0]

    # --- Logging ---

    print("\n=== Optimization finished using multi-start Adam ===\n")
    print(f"Duration: {duration:.3f} seconds on {devices}")
    print(f"Steps taken: {steps_taken}")
    print(f"Convergences: {converged_initializations}")

    # Show slowest converged gate
    if num_converged > 1:
        slowest_idx = jnp.argmax(durations_converged)
        slowest_duration = durations_converged[slowest_idx]
        slowest_infidelity = final_infidelities[converged][slowest_idx]
        slowest_params = jax.tree.map(float, unravel(params_converged[slowest_idx]))

        print("\nSlowest converged gate:")
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
        print("\nFastest converged gate:")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, num_converged, size=(1024, num_converged))
        mins = np.asarray(durations_converged)[idx].min(axis=1)
        duration_err = mins.std()
        print(
            f"> duration = {fastest_duration} (one-sided bootstrap error: {duration_err:.1g})"
        )
    else:
        print("Converged gate:")
        print(f"> duration = {fastest_duration}")

    print(f"> parameters = ({', '.join(str(p) for p in fastest_params)})")
    if float(fastest_infidelity) < 0:
        print("> infidelity <= numerical precision")
    else:
        print(f"> infidelity = {fastest_infidelity:.6e}")

    if return_all_converged:
        sorter = jnp.argsort(durations_converged)
        return [jax.tree.map(float, unravel(params_converged[i])) for i in sorter]

    return fastest_params
