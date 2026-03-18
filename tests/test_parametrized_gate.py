import jax
import jax.numpy as jnp
import pytest

import rydopt as ro
from rydopt.protocols import GateSystem, PulseAnsatzLike
from rydopt.pulses import MappedPulseAnsatz
from rydopt.types import GenericPulseParams, PulseParams


@pytest.fixture
def simple_gate() -> GateSystem:
    return ro.gates.TwoQubitGate(
        phi=None,
        theta=2 * jnp.pi * 0.2,
        Vnn=20.0,
        decay=0.0,
    )


@pytest.fixture
def pulse() -> MappedPulseAnsatz:
    degrees = [1, 0, 2, 0]
    num_params = [1, 1, 6, 1]

    pulse_map = ro.pulses.PolynomialPulseMap(
        degrees=degrees,
        num_params=num_params,
    )

    return ro.pulses.MappedPulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=ro.pulses.const,
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
    simple_gate: GateSystem, pulse: MappedPulseAnsatz, params: GenericPulseParams
) -> None:
    duration, detuning, phase, rabi = pulse.generate_pulse_params(simple_gate, params)

    detuning = jnp.asarray(detuning)
    phase = jnp.asarray(phase)
    rabi = jnp.asarray(rabi)

    assert jnp.isscalar(duration)
    assert detuning.shape == (1,)
    assert phase.shape == (6,)
    assert rabi.shape == (1,)
    assert jnp.asarray(duration) > 0


def test_evaluate_pulse_functions_real(
    simple_gate: GateSystem, pulse: MappedPulseAnsatz, params: GenericPulseParams
) -> None:
    t = jnp.linspace(0.0, 1.0, 10)

    d0, detuning, phase, rabi = pulse.evaluate_pulse_functions_for_gate(t, params, simple_gate)

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

    gates = [
        ro.gates.TwoQubitGate(
            phi=None,
            theta=4 * phase,
            Vnn=20.0,
            decay=0.0,
        )
        for phase in phases
    ]

    gate_family = ro.gates.ParametrizedGate(
        gates=gates,
        reduction="mean",
    )

    fid = gate_family.fidelity(
        pulse,
        params,
        tol=1e-3,
    )

    assert jnp.isfinite(fid)
    assert 0.0 <= fid <= 1.0
