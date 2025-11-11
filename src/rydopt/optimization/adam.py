import jax
import optax
import time
from rydopt.gates.fidelity import process_fidelity_from_states
from rydopt.optimization.opt_step import opt_step, loss_fn


# optmimization of a single pulse starting from given initial parameters
def adam(
    gate,
    pulse,
    params,
    T_penalty,
    N_epochs,
    learning_rate,
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
