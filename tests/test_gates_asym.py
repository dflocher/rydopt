import warnings

import numpy as np

import rydopt as ro

# Large finite interaction strength to approximate the blockade regime because float("inf")
# is not allowed in TwoQubitGateAsym, ThreeQubitGateAsym, FourQubitGateAsym.
V_LARGE = 50.0


def test_two_qubit_gate_asym_vs_symmetric() -> None:
    # Reference gate
    gate_sym = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=V_LARGE, decay=0)

    # Asymmetric gate with equal parameters to make it symmetric
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate_asym = ro.gates.TwoQubitGateAsym(phi1=None, phi2=None, theta12=np.pi, V12=V_LARGE, decay=0, s1=1.0, s2=1.0)

    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

    # Optimized parameters from test_cz.py
    params = (7.61141034, [-0.07884777], [1.83253308, -0.61765787], [])

    f_sym = ro.simulation.process_fidelity(gate_sym, pulse, params)
    f_asym = ro.simulation.process_fidelity(gate_asym, pulse, params)

    assert np.isclose(f_sym, f_asym, atol=1e-6), f"fidelities differ: {f_sym} vs {f_asym}"


def test_three_qubit_gate_asym_vs_isosceles() -> None:
    # Reference gate
    gate_iso = ro.gates.ThreeQubitGateIsosceles(
        phi=None,
        theta=np.pi,
        theta_prime=np.pi,
        lamb=np.pi,
        Vnn=V_LARGE,
        Vnnn=V_LARGE,
        decay=0.0,
    )

    # Asymmetric gate with equal parameters to make it symmetric
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate_asym = ro.gates.ThreeQubitGateAsym(
            phi1=None,
            phi2=None,
            phi3=None,
            theta12=np.pi,
            theta13=np.pi,
            theta23=np.pi,
            lamb=np.pi,
            V12=V_LARGE,
            V13=V_LARGE,
            V23=V_LARGE,
            decay=0.0,
            s1=1.0,
            s2=1.0,
            s3=1.0,
        )

    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

    # Optimized parameters from test_ccz.py
    params = (
        10.99552491,
        (0.20352,),
        (0.43322811, -1.18878954, 1.10057937, -0.70670388, 1.16454156, -0.25082207),
        (),
    )

    f_iso = ro.simulation.process_fidelity(gate_iso, pulse, params)
    f_asym = ro.simulation.process_fidelity(gate_asym, pulse, params)

    assert np.isclose(f_iso, f_asym, atol=1e-6), f"fidelities differ: {f_iso} vs {f_asym}"


def test_four_qubit_gate_asym_vs_pyramidal() -> None:
    # Reference gate
    gate_pyr = ro.gates.FourQubitGatePyramidal(
        phi=None,
        theta=np.pi,
        theta_prime=np.pi,
        lamb=np.pi,
        lamb_prime=np.pi,
        kappa=np.pi,
        Vnn=V_LARGE,
        Vnnn=V_LARGE,
        decay=0.0,
    )

    # Asymmetric gate with equal parameters to make it symmetric
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate_asym = ro.gates.FourQubitGateAsym(
            phi1=None,
            phi2=None,
            phi3=None,
            phi4=None,
            theta12=np.pi,
            theta13=np.pi,
            theta14=np.pi,
            theta23=np.pi,
            theta24=np.pi,
            theta34=np.pi,
            lamb123=np.pi,
            lamb124=np.pi,
            lamb134=np.pi,
            lamb234=np.pi,
            mu=np.pi,
            V12=V_LARGE,
            V13=V_LARGE,
            V14=V_LARGE,
            V23=V_LARGE,
            V24=V_LARGE,
            V34=V_LARGE,
            decay=0.0,
            s1=1.0,
            s2=1.0,
            s3=1.0,
            s4=1.0,
        )

    pulse = ro.pulses.PulseAnsatz(detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab)

    # Optimized parameters from test_cccz.py
    params = (
        12.42436209,
        (0.09580844,),
        (
            1.01592733,
            -1.00783188,
            2.07969005,
            -0.80292432,
            0.71405035,
            -0.16563671,
            0.72792360,
            0.32205233,
        ),
        (),
    )

    f_pyr = ro.simulation.process_fidelity(gate_pyr, pulse, params)
    f_asym = ro.simulation.process_fidelity(gate_asym, pulse, params)

    assert np.isclose(f_pyr, f_asym, atol=1e-6), f"fidelities differ: {f_pyr} vs {f_asym}"
