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

FloatParams: TypeAlias = float | tuple[float, ...]
BoolParams: TypeAlias = bool | tuple[bool, ...]


def _infidelity(params, gate: Gate, pulse: PulseAnsatz, tol: float):
    final_states = evolve(gate, pulse, params, tol)
    return 1 - gate.process_fidelity(final_states)


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

    def cond(carry):
        _, _, _, _, step, converged_initializations = carry
        return (step < num_steps) & (
            converged_initializations < min_converged_initializations
        )

    def body(carry):
        _, params, _, opt_state, step, _ = carry

        infidelity, grads = infidelity_and_grad(params)
        converged_initializations = jnp.sum(infidelity <= tol)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        log_step = (
            (step % 10 == 0)
            | (step == num_steps - 1)
            | (converged_initializations >= min_converged_initializations)
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
            step + 1,
            converged_initializations,
        )

    carry = (
        initial_params,
        initial_params,
        jnp.zeros_like(tol, dtype=float),
        opt_state0,
        0,
        0,
    )
    final_params, _, final_infidelity, _, step, converged_initializations = (
        jax.lax.while_loop(cond, body, carry)
    )

    return final_params, final_infidelity, step, converged_initializations


def adam(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params: tuple[FloatParams, ...],
    fixed_initial_params: tuple[BoolParams, ...] | None = None,
    num_steps: int = 1000,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
) -> tuple[FloatParams, ...]:
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol)
    infidelity_and_grad = jax.value_and_grad(infidelity)

    if fixed_initial_params is None:
        optimizer = optax.adam(learning_rate)
    else:
        optimizer = optax.chain(
            optax.adam(learning_rate),
            optax.transforms.freeze(fixed_initial_params),
        )

    # --- Optimize parameters ---

    print("")

    t0 = time.perf_counter()
    final_params, infidelity, step, converged_initializations = _adam_runner(
        infidelity_and_grad=infidelity_and_grad,
        optimizer=optimizer,
        initial_params=initial_params,
        num_steps=num_steps,
        min_converged_initializations=1,
        tol=tol,
    )
    jax.block_until_ready(final_params)
    duration = time.perf_counter() - t0

    if converged_initializations == 0:
        raise RuntimeError("No convergence. Try increasing num_steps or relaxing tol.")

    final_params = jax.tree.map(float, final_params)

    # --- Logging ---

    print("\n=== Optimization finished using Adam ===\n")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Steps taken: {step}")
    print(f"Convergences: {converged_initializations}")

    # Show converged gate
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
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol)
    infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))

    if fixed_initial_params is None:
        optimizer = optax.adam(learning_rate)
    else:
        optimizer = optax.chain(
            optax.adam(learning_rate),
            optax.transforms.freeze(fixed_initial_params),
        )

    # --- Create initial parameter samples ---

    leaves_min, treedef_min = jax.tree.flatten(min_initial_params)
    leaves_max, treedef_max = jax.tree.flatten(max_initial_params)

    if treedef_min != treedef_max:
        raise ValueError(
            "min_initial_params and max_initial_params must have the same shape."
        )

    key = jax.random.PRNGKey(seed)
    subkeys = jax.random.split(key, len(leaves_min))

    sampled_leaves = [
        jax.random.uniform(
            k,
            shape=(num_initializations,),
            minval=m,
            maxval=M,
        )
        for m, M, k in zip(leaves_min, leaves_max, subkeys)
    ]

    initial_params = treedef_min.unflatten(sampled_leaves)
    tol_vec = jnp.full((num_initializations,), tol)

    # --- Optimize parameters ---

    print("")

    devices = jax.devices()
    if len(devices) > 1:
        mesh = jax.make_mesh((len(devices),), ("batch",))
        sharding = jsr.NamedSharding(mesh, jsr.PartitionSpec("batch"))
        initial_params = jax.device_put(initial_params, sharding)

    t0 = time.perf_counter()
    final_params, final_infidelities, step, converged_initializations = _adam_runner(
        infidelity_and_grad=infidelity_and_grad,
        optimizer=optimizer,
        initial_params=initial_params,
        num_steps=num_steps,
        min_converged_initializations=min_converged_initializations,
        tol=tol_vec,
    )
    jax.block_until_ready(final_params)
    duration = time.perf_counter() - t0

    if converged_initializations == 0:
        raise RuntimeError("No convergence. Try increasing num_steps or relaxing tol.")

    converged = final_infidelities <= tol
    num_converged = int(jnp.sum(converged))
    durations_converged = final_params[0][converged]

    # --- Logging ---

    print("\n=== Optimization finished using multi-start Adam ===\n")
    print(f"Duration: {duration:.3f} seconds on {jax.devices()}")
    print(f"Steps taken: {step}")
    print(f"Convergences: {converged_initializations}")

    # Show slowest converged gate
    if num_converged > 1:
        slowest_idx = jnp.argmax(durations_converged)
        slowest_duration = durations_converged[slowest_idx]
        slowest_infidelity = final_infidelities[converged][slowest_idx]
        slowest_params = jax.tree.map(
            lambda a: float(a[converged][slowest_idx]),
            final_params,
        )

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
    fastest_params = jax.tree.map(
        lambda a: float(a[converged][fastest_idx]),
        final_params,
    )

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
        return [
            jax.tree.map(lambda a: float(a[converged][i]), final_params) for i in sorter
        ]

    return fastest_params
