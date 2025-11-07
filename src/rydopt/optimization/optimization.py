import jax
import jax.numpy as jnp
import optax
import diffrax
import time
from functools import partial
from rydopt import hamiltonians

jax.config.update("jax_enable_x64", True)


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
    Hamiltonians, input_states, fidelity_fn = hamiltonians.get_subsystem_Hamiltonians(
        n_atoms, Vnn, Vnnn, theta, eps, lamb, delta, kappa, decay
    )
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    start = time.time()
    for i in range(N_epochs):
        params, opt_state, loss_value = _opt_step(
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
    final_loss = _loss_fn(
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
    Hamiltonians, input_states, fidelity_fn = hamiltonians.get_subsystem_Hamiltonians(
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
            params, opt_state, loss_value = _opt_step(
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
    Hamiltonians, input_states, fidelity_fn = hamiltonians.get_subsystem_Hamiltonians(
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
            params, opt_state, loss_value = _opt_step(
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


# optimization step; called internally
@partial(
    jax.jit,
    static_argnames=["optimizer", "pulse", "Hamiltonians", "fidelity_fn", "T_penalty"],
)
def _opt_step(
    params,
    opt_state,
    optimizer,
    pulse,
    Hamiltonians,
    input_states,
    fidelity_fn,
    T_penalty,
):
    loss_value, grads = jax.value_and_grad(_loss_fn, argnums=0)(
        params, pulse, Hamiltonians, input_states, fidelity_fn, T_penalty
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


# loss function during optimization; called internally
def _loss_fn(params, pulse, Hamiltonians, input_states, fidelity_fn, T_penalty):
    output_states = []
    for H, psi_in in zip(Hamiltonians, input_states):
        schroedinger, args = pulse(H, params)
        psi_out = _propagate(psi_in, schroedinger, args)
        output_states.append(psi_out)
    return fidelity_fn(output_states) + T_penalty * jnp.abs(params[0])


# time evolution of a quantum state; called internally
def _propagate(psi_initial, schroedinger_eq, args):
    times = args[0]
    T = times[-1]
    term = diffrax.ODETerm(schroedinger_eq)
    solver = diffrax.Tsit5()  # Dopri8
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8, jump_ts=times)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T,
        dt0=None,
        y0=psi_initial,
        args=args,
        stepsize_controller=stepsize_controller,
        max_steps=10000,
    )
    psi = sol.ys[0].reshape(len(sol.ys[0]), 1)
    return psi  # it would be enough to return just the 1st element of the vector
