import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rydopt as ro
from rydopt.optimization.optimize import _make_infidelity, _make_value_and_grad, _resolve_gradient_mode
from rydopt.protocols import PulseAnsatz
from rydopt.pulses import PulseParams
from rydopt.types import ParamsFloatLike


def test_gradient_mode_selection() -> None:
    assert _resolve_gradient_mode("auto", 32) == "forward"
    assert _resolve_gradient_mode("auto", 33) == "reverse"
    assert _resolve_gradient_mode("forward", 100) == "forward"
    assert _resolve_gradient_mode("reverse", 1) == "reverse"

    with pytest.raises(ValueError, match="gradient_mode"):
        _resolve_gradient_mode("invalid", 1)


@pytest.mark.optimization
def test_forward_and_reverse_gradients_match() -> None:
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.Const(), phase_ansatz=ro.pulses.SinCrab(2))
    params = np.asarray(ro.pulses.PulseParams(7.6, [0.1], [1.8, -0.6], []), dtype=np.float64)
    infidelity = _make_infidelity(gate, pulse, params, np.arange(params.size), tol=1e-7)

    # Compile reverse mode first to verify that the later forward-mode trace
    # selects its own Diffrax sensitivity method rather than reusing this one.
    reverse_value_and_grad = jax.jit(_make_value_and_grad(infidelity, "reverse"))
    forward_value_and_grad = jax.jit(_make_value_and_grad(infidelity, "forward"))
    reverse_value, reverse_grad = reverse_value_and_grad(params)
    forward_value, forward_grad = forward_value_and_grad(params)

    assert np.allclose(forward_value, reverse_value, rtol=1e-10, atol=1e-12)
    assert np.allclose(forward_grad, reverse_grad, rtol=1e-10, atol=1e-12)


@pytest.mark.optimization
def test_auto_gradient_mode_falls_back_to_reverse() -> None:
    @jax.custom_vjp
    def custom_cost(params: jax.Array) -> jax.Array:
        return jnp.sum(params**2)

    def custom_cost_fwd(params: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.sum(params**2), params

    def custom_cost_bwd(params: jax.Array, cotangent: jax.Array) -> tuple[jax.Array]:
        return (2 * params * cotangent,)

    custom_cost.defvjp(custom_cost_fwd, custom_cost_bwd)

    class CustomGate:
        def cost(self, pulse: PulseAnsatz, params: ParamsFloatLike, tol: float) -> jax.Array:
            del pulse, tol
            return custom_cost(jnp.asarray(params))

    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.Const())
    params = ro.pulses.PulseParams(1.0, [0.2], [], [])
    result = ro.optimization.optimize(CustomGate(), pulse, params, num_steps=1, verbose=True)

    assert np.isclose(result.infidelity, 1.04)


@pytest.mark.optimization
def test_adam() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.Const(), phase_ansatz=ro.pulses.SinCrab(2))

    # Initial parameters
    initial_params = ro.pulses.PulseParams(7.6, [0.1], [1.8, -0.6], [])

    # Run optimization
    r = ro.optimization.optimize(
        gate, pulse, initial_params, num_steps=200, tol=1e-7, return_history=True, verbose=True
    )

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params)
    assert np.allclose(abs(1 - fidelity), r.infidelity, rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=1e-7)
    assert r.infidelity == r.infidelity_history[-1]


@pytest.mark.optimization
def test_adam_decay() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0.01)

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.Const(), phase_ansatz=ro.pulses.SinCrab(2))

    # Initial parameters
    initial_params = ro.pulses.PulseParams(7.6, [0.1], [1.8, -0.6], [])

    # Run optimization
    r = ro.optimization.optimize(gate, pulse, initial_params, num_steps=200, tol=1e-7)

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params)
    assert np.allclose(abs(1 - fidelity), r.infidelity, rtol=1e-12)


@pytest.mark.optimization
def test_multi_start_adam() -> None:
    tol = 1e-4

    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=2.0, decay=0)

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.ConstCosCrab(3))

    # Parameter bounds for choosing random initial parameters
    min_initial_params = ro.pulses.PulseParams(6, [-2, -2, -2], [], [])
    max_initial_params = ro.pulses.PulseParams(9, [2, 2, 2], [], [])

    # Run optimization
    r = ro.optimization.multi_start_optimize(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=500,
        num_initializations=100,
        min_converged_initializations=2,
        tol=tol,
        return_all=True,
        num_processes=1,
        return_history=True,
    )

    # Verify the fidelities of the 'min_converged_initializations'
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params[0], tol=tol)
    assert np.allclose(abs(1 - fidelity), r.infidelity[0], rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=tol)
    assert r.infidelity[0] == r.infidelity_history[-1, 0]

    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params[1], tol=tol)
    assert np.allclose(abs(1 - fidelity), r.infidelity[1], rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=tol)
    assert r.infidelity[1] == r.infidelity_history[-1, 1]


@pytest.mark.optimization
def test_multi_start_adam_decay() -> None:
    tol = 1e-3

    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=2.0, decay=0.0005)

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.ConstCosCrab(3))

    # Parameter bounds for choosing random initial parameters
    min_initial_params = ro.pulses.PulseParams(6, [-2, -2, -2], [], [])
    max_initial_params = ro.pulses.PulseParams(9, [2, 2, 2], [], [])

    # Run optimization
    r = ro.optimization.multi_start_optimize(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=100,
        num_initializations=20,
        tol=tol,
        num_processes=1,
    )

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params, tol=tol)
    assert np.allclose(abs(1 - fidelity), r.infidelity, rtol=1e-12)


@pytest.mark.optimization
def test_fastest() -> None:
    tol = 1e-4

    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.Const(), phase_ansatz=ro.pulses.SinCrab(2))

    # Parameter bounds for choosing random initial parameters
    min_initial_params = ro.pulses.PulseParams(6, [-2], [-2, -2], [])
    max_initial_params = ro.pulses.PulseParams(9, [2], [2, 2], [])

    # Run optimization
    r = ro.optimization.multi_start_optimize(
        gate,
        pulse,
        min_initial_params,
        max_initial_params,
        num_steps=500,
        num_initializations=100,
        min_converged_initializations=20,
        num_processes=4,
        tol=tol,
        return_history=True,
    )

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params, tol=tol)
    assert np.allclose(abs(1 - fidelity), r.infidelity, rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=tol)
    assert r.infidelity == r.infidelity_history[-1]


@pytest.mark.optimization
def test_fixed() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.Const(), phase_ansatz=ro.pulses.SinCrab(2))

    # Initial parameters
    initial_params = ro.pulses.PulseParams(7.6, [0.0], [1.8, -0.6], [])
    fixed_initial_params = ro.pulses.PulseParams(False, [True], [False, False], [])

    # Run optimization
    r = ro.optimization.optimize(gate, pulse, initial_params, fixed_initial_params, num_steps=200)

    # Verify the fidelity
    fidelity = ro.simulation.process_fidelity(gate, pulse, r.params)
    assert np.allclose(abs(1 - fidelity), r.infidelity, rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=1e-7)


@pytest.mark.optimization
def test_adam_average_gate_fidelity() -> None:
    # Gate
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0, fidelity_type="average_gate")

    # Pulse
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.Const(), phase_ansatz=ro.pulses.SinCrab(2))

    # Initial parameters
    initial_params = ro.pulses.PulseParams(7.6, [0.1], [1.8, -0.6], [])

    # Run optimization using average gate fidelity
    r = ro.optimization.optimize(gate, pulse, initial_params, num_steps=200, tol=1e-7)

    # Verify the fidelity matches what the result reports
    fidelity = ro.simulation.average_gate_fidelity(gate, pulse, r.params)
    assert np.allclose(abs(1 - fidelity), r.infidelity, rtol=1e-12)
    assert np.allclose(fidelity, 1, rtol=1e-7)


@pytest.mark.optimization
def test_optimize_accepts_flat_params_like() -> None:
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)
    pulse = ro.pulses.SinglePhotonPulseAnsatz(detuning_ansatz=ro.pulses.Const(), phase_ansatz=ro.pulses.SinCrab(2))

    initial_params = [7.6, 0.1, 1.8, -0.6]
    fixed_initial_params = [False, True, False, False]

    result = ro.optimization.optimize(
        gate,
        pulse,
        initial_params,
        fixed_initial_params,
        num_steps=200,
        tol=1e-7,
    )

    assert isinstance(result.params, PulseParams)
    assert len(result.params) == 4
    assert np.isclose(result.params[1], initial_params[1])

    fidelity = ro.simulation.process_fidelity(gate, pulse, result.params)
    assert np.allclose(abs(1 - fidelity), result.infidelity, rtol=1e-12)
