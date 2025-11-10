import jax.numpy as jnp


def fidelity(self, final_states, target_states, multiplicities):
    overlaps = jnp.stack([jnp.vdot(t, f) for t, f in zip(target_states, final_states)])

    s = 1 + jnp.sum(overlaps * multiplicities)
    d = 1 + jnp.sum(multiplicities)

    return jnp.abs(s) ** 2 / d**2
