import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rydopt as ro
from rydopt.protocols import GateWithInterpolationParam, PulseAnsatzLike
from rydopt.pulses import PulseFamilyAnsatz
from rydopt.types import ParamsFloatLike, PulseParams


@pytest.fixture
def simple_gate_with_interpolation_param() -> GateWithInterpolationParam:
    gate = ro.gates.TwoQubitGate(
        phi=None,
        theta=2 * jnp.pi * 0.2,
        Vnn=20.0,
        decay=0.0,
    )
    interp_param = 0.2
    return GateWithInterpolationParam(gate, interp_param)


@pytest.fixture
def pulse() -> PulseFamilyAnsatz:
    degrees = [1, 0, 2, 0]
    num_params = 6

    pulse_map = ro.pulses.PolynomialPulseMap(
        degrees=degrees,
    )

    return ro.pulses.PulseFamilyAnsatz(
        detuning_ansatz=ro.pulses.Const(),
        phase_ansatz=ro.pulses.SinCrab(num_params),
        rabi_ansatz=ro.pulses.Const(),
        pulse_map=pulse_map,
    )


@pytest.fixture
def params() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # consistent with degrees above
    return (
        jnp.array([2.0, 1.0]),
        jnp.array([0.5]),
        jnp.ones((6, 3)),
        jnp.array([1.0]),
    )


def test_generate_pulse_params_real(
    simple_gate_with_interpolation_param: GateWithInterpolationParam,
    pulse: PulseFamilyAnsatz,
    params: ParamsFloatLike,
) -> None:
    duration, detuning, phase, rabi = pulse.generate_pulse_params(simple_gate_with_interpolation_param, params)

    detuning = jnp.asarray(detuning)
    phase = jnp.asarray(phase)
    rabi = jnp.asarray(rabi)

    assert jnp.isscalar(duration)
    assert detuning.shape == (1,)
    assert phase.shape == (6,)
    assert rabi.shape == (1,)
    assert jnp.asarray(duration) > 0


def test_evaluate_pulse_functions_real(
    simple_gate_with_interpolation_param: GateWithInterpolationParam,
    pulse: PulseFamilyAnsatz,
    params: PulseParams,
) -> None:
    t = jnp.linspace(0.0, 1.0, 10)

    d0, detuning, phase, rabi = pulse.evaluate_pulse_functions_for_gate(t, params, simple_gate_with_interpolation_param)

    detuning = jnp.asarray(detuning)
    phase = jnp.asarray(phase)
    rabi = jnp.asarray(rabi)

    assert d0.shape == t.shape
    assert detuning.shape == t.shape
    assert phase.shape == t.shape
    assert rabi.shape == t.shape

    assert not jnp.isnan(detuning).any()
    assert not jnp.isnan(phase).any()
    assert not jnp.isnan(rabi).any()


def test_parametrized_gate_real(pulse: PulseAnsatzLike, params: PulseParams) -> None:
    phases = jnp.array([0.1, 0.15, 0.2]) * jnp.pi

    sampled_gates = [
        ro.gates.TwoQubitGate(
            phi=None,
            theta=4 * phase,
            Vnn=20.0,
            decay=0.0,
        )
        for phase in phases
    ]

    parametrized_gate = ro.gates.ParametrizedGate(
        fixed_parameter_gates=sampled_gates,
        parameter_values=phases,
        reduction="mean",
    )

    infid = parametrized_gate.cost(
        pulse,
        params,
        tol=1e-3,
    )

    assert jnp.isfinite(infid)
    assert 0.0 <= infid <= 1.0


@pytest.mark.optimization
def test_cphase() -> None:
    # target phases
    phases = jnp.array([0.1, 0.2]) * jnp.pi

    # parametrized gate
    sampled_gates = [
        ro.gates.TwoQubitGate(
            phi=None,
            theta=4 * phase,
            Vnn=20.0,
            decay=0.0,
        )
        for phase in phases
    ]

    parametrized_gate = ro.gates.ParametrizedGate(
        fixed_parameter_gates=sampled_gates,
        parameter_values=phases,
        reduction="mean",
    )

    # Pulse
    degrees = [0, 0, 1, 0]
    n_params = 6
    pulse_mal = ro.pulses.PolynomialPulseMap(degrees=degrees)
    pulse_family = ro.pulses.PulseFamilyAnsatz(
        detuning_ansatz=ro.pulses.Const(),
        phase_ansatz=ro.pulses.SinCrab(n_params),
        pulse_map=pulse_mal,
    )

    # Initial parameters
    initial_params = ro.pulses.PulseParams(15.0, [0.1], np.zeros((n_params, degrees[2] + 1)), [])

    # Run optimization
    r = ro.optimization.optimize(parametrized_gate, pulse_family, initial_params, num_steps=100, tol=1e-6)
    duration, detuning, phase, rabi = r.params
    detuning = jnp.asarray(detuning)
    phase = jnp.asarray(phase)
    rabi = jnp.asarray(rabi)

    assert phase.shape == (n_params, degrees[2] + 1)
    assert detuning.shape == (1,)
    assert rabi.shape == (0,)
    assert isinstance(duration, jax.Array)
    assert 0.0 <= r.infidelity <= 1e-2
