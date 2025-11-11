import jax
import jax.numpy as jnp
import optax
import time
from rydopt import gates
from rydopt.optimization.opt_step import opt_step, loss_fn


# optmimization of a single pulse starting from given initial parameters
def train_single_gate(
    n_atoms,
    Vnn,
    Vnnn,
    theta,
    eps,
    lamb,
    delta,
    kappa,
    pulse,
    params,
    N_epochs,
    learning_rate,
    T_penalty,
    decay,
):
    Hamiltonians, input_states, fidelity_fn = gates.get_subsystem_Hamiltonians(
        n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, decay
    )
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    start = time.time()
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
        if i % 10 == 0:
            print("{loss:.6f}".format(loss=loss_value))
            # print('{loss:.4e}'.format(loss=loss_value+1))
    end = time.time()
    final_params_str = "[" + ", ".join("{p:.8f}".format(p=p) for p in params) + "]"
    final_loss = loss_fn(
        params, pulse, Hamiltonians, input_states, fidelity_fn, T_penalty
    )
    print("Training time:     {t:.3f} s".format(t=end - start))
    print("Final parameters: ", final_params_str)
    print("Final loss:       {loss:.6f}".format(loss=final_loss))
    print("Final loss + 1:    {loss:.4e} \n".format(loss=final_loss + 1))
    return params


# optimization of multiple pulses from random initial parameters
def gate_search(
    n_atoms,
    Vnn,
    Vnnn,
    theta,
    eps,
    lamb,
    delta,
    kappa,
    pulse,
    T_default,
    N_searches,
    N_params,
    N_epochs,
    learning_rate,
    T_penalty,
    decay,
):
    Hamiltonians, input_states, fidelity_fn = gates.get_subsystem_Hamiltonians(
        n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, decay
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


# optimization of multiple pulses from random initial parameters. No print statements. Called from 'cluster_gate_optimization.py'
def gate_search_cluster(
    n_atoms,
    Vnn,
    Vnnn,
    theta,
    eps,
    lamb,
    delta,
    kappa,
    pulse,
    T_default,
    N_searches,
    N_params,
    N_epochs,
    learning_rate,
    T_penalty,
    decay,
):
    Hamiltonians, input_states, fidelity_fn = gates.get_subsystem_Hamiltonians(
        n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, decay
    )
    optimizer = optax.adam(learning_rate=learning_rate)
    key = jax.random.PRNGKey(time.time_ns())
    all_costs = jnp.zeros((N_searches))
    all_params = jnp.zeros((N_searches, N_params))
    start = 0.0
    for j in range(N_searches):
        if (
            j == 1
        ):  # start measuring the runtime from the 2nd search to avoid measuring the compile time
            start = time.time()
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
        all_costs = all_costs.at[j].set(loss_value)
        all_params = all_params.at[j, :].set(params)
    end = time.time()
    runtime = end - start
    return all_costs, all_params, runtime
