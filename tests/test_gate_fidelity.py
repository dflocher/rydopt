import jax
import jax.numpy as jnp
import numpy as np

import rydopt as ro


def test_two_qubit_gate_fidelity():
    # Create a gate
    phi = 0.2
    theta = 0.3
    gate = ro.gates.TwoQubitGate(phi=phi, theta=theta, Vnn=float("inf"), decay=0)

    # Create a jitted process fidelity function for the created gate
    fidelity_jit = jax.jit(lambda final_basis_states: gate.process_fidelity(final_basis_states))

    # Create the expected final states
    target_states = (
        jnp.array([jnp.exp(1j * phi), 0.0 + 0.0j]),
        jnp.array([jnp.exp(1j * (2 * phi + theta)), 0.0 + 0.0j]),
    )

    # --- Perfect implementation ---
    f = fidelity_jit(target_states)
    ref = 1.0
    assert np.isclose(f, ref)

    # --- Extra single-qubit Z phase on |01> ---
    delta = 0.1
    finals = (jnp.exp(1j * delta) * target_states[0], target_states[1])
    f = fidelity_jit(finals)
    ref = np.cos(delta / 2) ** 2
    assert np.isclose(f, ref)

    # --- Leakage out of the computational subspace ---
    amp10, amp11 = 0.95, 0.90
    leak10 = jnp.sqrt(1.0 - amp10**2)
    leak11 = jnp.sqrt(1.0 - amp11**2)
    finals = (
        jnp.array([amp10 * jnp.exp(1j * phi), leak10]),
        jnp.array([amp11 * jnp.exp(1j * (2 * phi + theta)), leak11]),
    )
    f = fidelity_jit(finals)
    ref = (1 + 2 * amp10 + amp11) ** 2 / 16
    assert np.isclose(f, ref)


def test_two_qubit_gate_fidelity_free_phi():
    # Create a gate
    phi = None
    theta = 0.3
    gate = ro.gates.TwoQubitGate(phi=phi, theta=theta, Vnn=float("inf"), decay=0)

    # Create a jitted process fidelity function for the created gate
    fidelity_jit = jax.jit(lambda final_basis_states: gate.process_fidelity(final_basis_states))

    # Create the expected final states
    target_states = (
        jnp.array([jnp.exp(1j * 0), 0.0 + 0.0j]),
        jnp.array([jnp.exp(1j * (2 * 0 + theta)), 0.0 + 0.0j]),
    )

    # --- Single-qubit Z phase on all atoms ---
    delta = 0.37
    finals = (
        jnp.exp(1j * delta) * target_states[0],
        jnp.exp(1j * 2 * delta) * target_states[1],
    )
    f = fidelity_jit(finals)
    assert np.isclose(f, 1.0)


def test_two_qubit_gate_fidelity_free_theta():
    # Create a gate with free theta
    phi = 0.2
    theta = None
    gate = ro.gates.TwoQubitGate(phi=phi, theta=theta, Vnn=float("inf"), decay=0)

    # Create a jitted process fidelity function for the created gate
    fidelity_jit = jax.jit(lambda final_basis_states: gate.process_fidelity(final_basis_states))

    # Create the expected final states
    target_states = (
        jnp.array([jnp.exp(1j * phi), 0.0 + 0.0j]),
        jnp.array([jnp.exp(1j * (2 * phi + 0)), 0.0 + 0.0j]),
    )

    # --- Controlled-phase shift on |11> ---
    delta = 0.53
    finals = (
        target_states[0],
        jnp.exp(1j * delta) * target_states[1],
    )
    f = fidelity_jit(finals)
    assert np.isclose(f, 1.0)
