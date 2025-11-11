import jax
import jax.numpy as jnp
import optax
import time
from rydopt.gates.fidelity import process_fidelity_from_states
from rydopt.optimization.opt_step import opt_step


# optimization of multiple pulses from random initial parameters
def multi_start_adam(
    gate,
    pulse,
    T_default,
    N_searches,
    N_params,
    N_epochs,
    learning_rate,
    T_penalty,
):
    Hamiltonians = gate.subsystem_hamiltonians()

    input_states = gate.initial_states()

    target_states = gate.target_states()
    multiplicities = gate.multiplicities()
    eliminate = gate.phase_eliminator()
    fidelity_fn = jax.jit(
        lambda final_states: -process_fidelity_from_states(
            final_states, target_states, multiplicities, eliminate
        )
    )

    optimizer = optax.adam(learning_rate=learning_rate)
    key = jax.random.PRNGKey(time.time_ns())
    best_loss = 0.0
    best_params = None
    best_index = None
    for j in range(N_searches):
        key, subkey = jax.random.split(key)
        r = jax.random.normal(subkey, (N_params,))
        params = jnp.array(
            [T_default + 0.3 * T_default * r[0]] + [r[i] for i in range(1, N_params)]
        )
        opt_state = optimizer.init(params)
        loss_value = 0.0
        for i in range(N_epochs):
            params, opt_state, loss_value = opt_step(
                params,
                opt_state,
                optimizer,
                pulse,
                Hamiltonians,
                input_states,
                fidelity_fn,
                T_penalty,
            )
        result_str = "search: {search:d}, loss: {loss:.6f}, fidelity: {fid:.6f}".format(
            search=j, loss=loss_value, fid=loss_value - T_penalty * params[0]
        )
        if loss_value - T_penalty * params[0] <= -0.999:
            params_str = (
                ", params: [" + ", ".join("{p:.5f}".format(p=p) for p in params) + "]"
            )
            print(result_str + params_str)
        else:
            print(result_str)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = params
            best_index = j
    print(
        "\nBest run: {r:d}, cost: {c:.6f}, T: {T:.4f}".format(
            r=best_index, c=best_loss, T=best_params[0]
        )
    )
    return best_params
