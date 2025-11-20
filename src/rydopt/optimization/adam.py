from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation import evolve
from functools import partial
import jax.numpy as jnp
import jax.sharding as jshard
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


@partial(
    jax.jit,
    static_argnames=[
        "infidelity_and_grad",
        "optimizer",
        "axis_name",
        "num_steps",
        "min_converged_initializations",
    ],
    donate_argnames=["initial_params"],
)
def _adam_runner(
    infidelity_and_grad,
    optimizer: optax.GradientTransformation,
    initial_params,
    axis_name: str | None,
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
                "Step {step:06d} on device {idx:03d}: min infidelity ={infidelity:13.6e}, converged = {converged}",
                step=step,
                idx=jax.lax.axis_index(axis_name) if axis_name else 0,
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
            optax.transforms.freeze(flat_fixed),
        )

    # Infidelity and its gradient
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol, unravel=unravel)
    infidelity_and_grad = jax.value_and_grad(infidelity)

    # --- Optimize parameters ---

    print("")

    t0 = time.perf_counter()
    final_params, final_infidelity = _adam_runner(
        infidelity_and_grad=infidelity_and_grad,
        optimizer=optimizer,
        initial_params=flat_params,
        axis_name=None,
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
    num_devices = len(jax.devices())
    axis_name = "batch" if num_devices > 1 else None

    # Pad the number of initial parameter samples to be a multiple of the number of devices
    num_initializations += (-num_initializations) % num_devices

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
        flat_fixed, _ = ravel_pytree(fixed_initial_params)
        optimizer = optax.chain(
            optax.adam(learning_rate),
            optax.transforms.freeze(flat_fixed),
        )

    # Infidelity and its gradient
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol, unravel=unravel)
    infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))

    # Function that runs the optimization
    def runner(flat_params, tol):
        return _adam_runner(
            infidelity_and_grad=infidelity_and_grad,
            optimizer=optimizer,
            initial_params=flat_params,
            axis_name=axis_name,
            num_steps=num_steps,
            min_converged_initializations=(
                min_converged_initializations + num_devices - 1
            )
            // num_devices,
            tol=tol,
        )

    if num_devices > 1:
        mesh = jax.make_mesh(
            (num_devices,), (axis_name,), axis_types=(jshard.AxisType.Auto,)
        )
        spec_batch = jshard.PartitionSpec(axis_name)
        sharding = jax.NamedSharding(mesh, spec_batch)
        flat_params = jax.device_put(flat_params, sharding)
        tol_vec = jax.device_put(tol_vec, sharding)
        runner = jax.shard_map(
            runner,
            mesh=mesh,
            in_specs=(spec_batch, spec_batch),
            out_specs=(spec_batch, spec_batch),
            check_vma=False,
        )

    # --- Optimize parameters ---

    print("")

    t0 = time.perf_counter()
    final_params, final_infidelities = runner(flat_params, tol_vec)
    final_params, final_infidelities = jax.device_get(
        (final_params, final_infidelities)
    )
    duration = time.perf_counter() - t0

    converged = final_infidelities <= tol
    num_converged = int(jnp.sum(converged))
    if num_converged == 0:
        converged = [jnp.argmin(final_infidelities)]
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
