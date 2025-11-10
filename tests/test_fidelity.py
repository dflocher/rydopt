import rydopt as ro
import jax
import jax.numpy as jnp
import numpy as np


def test_two_qubit_gate_fidelity():
    # Create a gate
    phi = 0.2
    theta = 0.3
    gate = ro.gates.TwoQubitGate(phi=phi, theta=theta, Vnn=float("inf"), decay=0)

    # Create a jitted process fidelity function for the created gate
    target_states = gate.target_states()
    multiplicities = gate.multiplicities()
    fidelity_jit = jax.jit(
        lambda final_states: ro.gates.fidelity(
            final_states, target_states, multiplicities
        )
    )

    # --- Perfect implementation ---
    f = fidelity_jit(target_states)
    ref = 1.0
    assert np.isclose(f, ref)

    # --- Extra single-qubit Z phase on |01> ---
    delta = 0.1
    finals_phase = (jnp.exp(1j * delta) * target_states[0], target_states[1])
    f = fidelity_jit(finals_phase)
    ref = np.cos(delta / 2) ** 2
    assert np.isclose(f, ref)

    # --- Leakage out of the computational subspace ---
    amp10, amp11 = 0.95, 0.90
    leak10 = jnp.sqrt(1.0 - amp10**2)
    leak11 = jnp.sqrt(1.0 - amp11**2)
    finals_leak = (
        jnp.array([amp10 * jnp.exp(1j * phi), leak10]),
        jnp.array([amp11 * jnp.exp(1j * (2 * phi + theta)), leak11]),
    )
    f = fidelity_jit(finals_leak)
    ref = (1 + 2 * amp10 + amp11) ** 2 / 16
    assert np.isclose(f, ref)
