import jax.numpy as jnp


def process_fidelity(final_states, target_states, multiplicities, eliminate_phase=None):
    overlaps = jnp.stack([jnp.vdot(t, f) for t, f in zip(target_states, final_states)])

    if eliminate_phase is not None:
        overlaps = eliminate_phase(overlaps)

    s = 1 + jnp.sum(overlaps * multiplicities)
    d = 1 + jnp.sum(multiplicities)

    return jnp.abs(s) ** 2 / d**2
