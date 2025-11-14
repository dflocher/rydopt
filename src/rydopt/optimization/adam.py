from rydopt.gates.gate import Gate
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation import evolve
from functools import partial
import jax.numpy as jnp
import jax
import time
import optax


def _infidelity(params, gate: Gate, pulse: PulseAnsatz, tol: float):
    final_states = evolve(gate, pulse, params, tol)
    return 1 - gate.process_fidelity(final_states)


@partial(
    jax.jit,
    static_argnames=[
        "infidelity_and_grad",
        "optimizer",
        "num_steps",
    ],
    donate_argnames=["initial_params"],
)
def _adam_runner(
    infidelity_and_grad,
    optimizer: optax.GradientTransformation,
    initial_params,
    num_steps: int,
    tol,
):
    opt_state0 = optimizer.init(initial_params)

    def body(carry, step):
        params, opt_state, was_done, steps_taken = carry

        def do_step(_):
            infidelity, grads = infidelity_and_grad(params)
            done = jnp.any(was_done | (infidelity <= tol))
            new_steps_taken = steps_taken + 1

            def apply_update(_):
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state

            def keep_params(_):
                return params, opt_state

            new_params, new_opt_state = jax.lax.cond(
                done,
                keep_params,
                apply_update,
                operand=None,
            )

            return (new_params, new_opt_state, done, new_steps_taken), infidelity

        def do_no_step(_):
            return carry, jnp.zeros_like(tol)

        (params, opt_state, done, steps_taken), infidelity = jax.lax.cond(
            was_done,
            do_no_step,
            do_step,
            operand=None,
        )

        is_periodic = (step % 10 == 0) | (step == num_steps - 1)
        log_step = (~was_done) & (is_periodic | done)

        jax.lax.cond(
            jnp.any(log_step),
            lambda _: jax.debug.print(
                "Step {step:05d}: infidelity = {infidelity:13.6e}",
                step=step,
                infidelity=jnp.min(infidelity),
            ),
            lambda _: None,
            operand=None,
        )

        return (params, opt_state, done, steps_taken), infidelity

    (final_params, _, _, steps_taken), infidelity_history = jax.lax.scan(
        body,
        (initial_params, opt_state0, False, 0),
        jnp.arange(num_steps),
    )

    return final_params, infidelity_history, steps_taken


def adam(
    gate: Gate,
    pulse: PulseAnsatz,
    initial_params,
    num_steps: int,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
):
    initial_params = tuple(jnp.asarray(p) for p in initial_params)

    # Optimize parameters
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=0.1 * tol)
    infidelity_and_grad = jax.value_and_grad(infidelity)

    optimizer = optax.adam(learning_rate)

    print("")

    t0 = time.perf_counter()
    final_params, infidelity_history, steps_taken = _adam_runner(
        infidelity_and_grad=infidelity_and_grad,
        optimizer=optimizer,
        initial_params=initial_params,
        num_steps=num_steps,
        tol=tol,
    )
    duration = time.perf_counter() - t0

    # Show results
    print("\n=== Optimization finished using Adam ===")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Steps taken: {steps_taken}")
    print(f"Final parameters: ({', '.join(str(p) for p in final_params)})")
    print(f"Final infidelity: {infidelity_history[steps_taken - 1]:.6e}")

    return final_params


def multi_start_adam(
    gate: Gate,
    pulse: PulseAnsatz,
    min_initial_params,
    max_initial_params,
    num_starts: int,
    num_steps: int,
    learning_rate: float = 0.05,
    tol: float = 1e-7,
    seed: int = 0,
):
    min_initial_params = tuple(jnp.asarray(p) for p in min_initial_params)
    max_initial_params = tuple(jnp.asarray(p) for p in max_initial_params)

    # Create initial parameter samples
    leaves_min, treedef_min = jax.tree.flatten(min_initial_params)
    leaves_max, treedef_max = jax.tree.flatten(max_initial_params)

    if treedef_min != treedef_max:
        raise ValueError(
            "min_initial_params and max_initial_params must have the same shape."
        )

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, len(leaves_min))

    sampled_leaves = []
    for m, M, k in zip(leaves_min, leaves_max, keys):
        sampled = jax.random.uniform(
            k,
            shape=(num_starts,) + m.shape,
            minval=m,
            maxval=M,
        )
        sampled_leaves.append(sampled)

    initial_params = treedef_min.unflatten(sampled_leaves)
    tol_vec = jnp.full((num_starts,), tol)

    # Optimize parameters
    infidelity = partial(_infidelity, gate=gate, pulse=pulse, tol=tol)
    infidelity_and_grad = jax.vmap(jax.value_and_grad(infidelity))

    optimizer = optax.adam(learning_rate)

    print("")

    t0 = time.perf_counter()
    final_params, infidelity_history, steps_taken = _adam_runner(
        infidelity_and_grad=infidelity_and_grad,
        optimizer=optimizer,
        initial_params=initial_params,
        num_steps=num_steps,
        tol=tol_vec,
    )
    duration = time.perf_counter() - t0

    final_infidelities = infidelity_history[steps_taken - 1]
    best_idx = jnp.argmin(final_infidelities)
    best_infidelity = final_infidelities[best_idx]
    best_params = jax.tree.map(lambda x: x[best_idx], final_params)

    # Show results
    print("\n=== Optimization finished using multi-start Adam ===")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Steps taken: {steps_taken}")
    print(f"Final parameters: ({', '.join(str(p) for p in best_params)})")
    print(f"Final infidelity: {best_infidelity:.6e}")

    return best_params
