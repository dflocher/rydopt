import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rydopt as ro
from rydopt.gates import GateFamily
from rydopt.pulses import PulseFamilyAnsatz
from rydopt.types import ParamsFloatLike, PulseParams


@pytest.fixture
def simple_gate_family() -> GateFamily:
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

    return ro.gates.GateFamily(
        fixed_parameter_gates=sampled_gates,
        parameter_values=phases,
        reduction="mean",
    )


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
    simple_gate_family: GateFamily,
    pulse: PulseFamilyAnsatz,
    params: ParamsFloatLike,
) -> None:
    gate_param = simple_gate_family.parameter_values[0]
    duration, detuning, phase, rabi = pulse._generate_pulse_params_arrays(params, gate_param)

    assert jnp.ndim(duration) == 0
    assert detuning.shape == (1,)
    assert phase.shape == (6,)
    assert rabi.shape == (1,)
    assert jnp.asarray(duration) > 0


def test_evaluate_pulse_functions_real(
    simple_gate_family: GateFamily,
    pulse: PulseFamilyAnsatz,
    params: PulseParams,
) -> None:
    t = jnp.linspace(0.0, 1.0, 10)
    for gate_param in simple_gate_family.parameter_values:
        d0, detuning, phase, rabi = pulse.evaluate_pulse_functions(t, params, gate_param)

        assert d0.shape == t.shape
        assert detuning.shape == t.shape
        assert phase.shape == t.shape
        assert rabi.shape == t.shape

        assert not jnp.isnan(detuning).any()
        assert not jnp.isnan(phase).any()
        assert not jnp.isnan(rabi).any()


def test_gate_family_real(pulse: PulseFamilyAnsatz, params: PulseParams) -> None:
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

    gate_family = ro.gates.GateFamily(
        fixed_parameter_gates=sampled_gates,
        parameter_values=phases,
        reduction="mean",
    )

    infid = gate_family.cost(
        pulse,
        params,
        tol=1e-3,
    )

    assert jnp.isfinite(infid)
    assert 0.0 <= infid <= 1.0


@pytest.mark.optimization
def test_cphase() -> None:
    # target phases
    phases = jnp.array([0.1, 0.25]) * jnp.pi

    # gate family
    sampled_gates = [
        ro.gates.TwoQubitGate(
            phi=None,
            theta=4 * phase,
            Vnn=20.0,
            decay=0.0001,
        )
        for phase in phases
    ]

    gate_family = ro.gates.GateFamily(
        fixed_parameter_gates=sampled_gates,
        parameter_values=phases,
        reduction="mean",
    )

    # Pulse
    degrees = [1, 0, 2, 0]
    n_params = 10
    pulse_mal = ro.pulses.PolynomialPulseMap(degrees=degrees)
    pulse_family = ro.pulses.PulseFamilyAnsatz(
        detuning_ansatz=ro.pulses.Const(),
        phase_ansatz=ro.pulses.SinCrab(n_params),
        pulse_map=pulse_mal,
    )

    # Initial parameters
    initial_params = ro.pulses.PulseFamilyParams(
        8.0 * np.ones(degrees[0] + 1), 0.1 * np.ones(degrees[1] + 1), np.ones((n_params, degrees[2] + 1)), []
    )

    # Run optimization
    r = ro.optimization.optimize(gate_family, pulse_family, initial_params, num_steps=200, tol=1e-6)
    duration, detuning, phase, rabi = r.params

    assert isinstance(duration, np.ndarray)
    assert duration.shape == (degrees[0] + 1,)
    assert isinstance(phase, np.ndarray)
    assert phase.shape == (n_params * (degrees[2] + 1),)
    assert isinstance(detuning, np.ndarray)
    assert detuning.shape == (degrees[1] + 1,)
    assert isinstance(rabi, np.ndarray)
    assert rabi.shape == (0,)
    assert 0.0 <= r.infidelity <= 1e-3

    pulse = pulse_family.pulse_ansatz
    for gate, value in zip(gate_family.gates, gate_family.parameter_values):
        params = pulse_family.generate_pulse_params(r.params, value)
        infidelity, infidelity_nodecay, ryd_time = ro.characterization.analyze_gate(
            gate,
            pulse,
            params,
            tol=1e-8,
        )

        assert isinstance(infidelity, float)
        assert infidelity <= 1e-3
        assert isinstance(infidelity_nodecay, float)
        assert infidelity_nodecay <= infidelity
        assert isinstance(ryd_time, float)
        assert 0.0 <= ryd_time <= params.duration
