import jax
import jax.numpy as jnp
import optax
import diffrax
from functools import partial


# optimization step; called internally
@partial(
    jax.jit,
    static_argnames=["optimizer", "pulse", "Hamiltonians", "fidelity_fn", "T_penalty"],
)
def opt_step(
    params,
    opt_state,
    optimizer,
    pulse,
    Hamiltonians,
    input_states,
    fidelity_fn,
    T_penalty,
):
    loss_value, grads = jax.value_and_grad(loss_fn, argnums=0)(
        params, pulse, Hamiltonians, input_states, fidelity_fn, T_penalty
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


# loss function during optimization; called internally
def loss_fn(params, pulse, Hamiltonians, input_states, fidelity_fn, T_penalty):
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
    psi = sol.ys[0].reshape(-1)
    return psi  # it would be enough to return just the 1st element of the vector
