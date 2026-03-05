import jax.numpy as jnp
import numpy as np
import pytest
import qutip as qt
from scipy.interpolate import interp1d

import rydopt as ro


def test_effective_controls() -> None:
    duration = 7.0
    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const_cos_crab,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=ro.pulses.const,
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const_sin_crab,
    )

    lower_params = (duration, [-1.6, 0.4, -0.2], [0.7, -0.3], [2.2])
    upper_params = (duration, [-0.5], [0.1], [1.4, 0.2, -0.1])
    packed_params = (
        duration,
        np.array([*lower_params[1], *upper_params[1]]),
        np.array([*lower_params[2], *upper_params[2]]),
        np.array([*lower_params[3], *upper_params[3]]),
    )

    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(len(lower_params[1]), len(lower_params[2]), len(lower_params[3])),
    )

    times = jnp.linspace(0.0, duration, 13)
    detuning, phase, rabi = pulse.evaluate_pulse_functions(times, packed_params)

    lower_detuning, lower_phase, lower_rabi = lower.evaluate_pulse_functions(times, lower_params)
    upper_detuning, upper_phase, upper_rabi = upper.evaluate_pulse_functions(times, upper_params)
    expected_detuning = lower_detuning + upper_detuning + (lower_rabi**2 - upper_rabi**2) / (4.0 * lower_detuning)
    expected_phase = lower_phase + upper_phase
    expected_rabi = lower_rabi * upper_rabi / (2.0 * lower_detuning)

    assert np.allclose(detuning, expected_detuning)
    assert np.allclose(phase, expected_phase)
    assert np.allclose(rabi, expected_rabi)


@pytest.mark.optimization
def test_two_photon_cz() -> None:
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)

    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=ro.pulses.const,
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const,
    )
    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(1, 2, 1),
        decay=0,
    )

    initial_params = (7.6, [-50.0, 50.0], [1.8, -0.6], [10.0, 10.0])  # duration, detuning, phase, rabi

    # Parameters of the upper transition and Rabi frequencies are fixed
    fixed_initial_params = (False, [False, True], [False, False], [True, True])

    result = ro.optimization.optimize(gate, pulse, initial_params, fixed_initial_params, num_steps=200, tol=1e-7)

    ref = (
        7.600019896010689,
        [-49.92218101, 50],
        [1.75873066, -0.61830304],
        [10, 10],
    )
    assert all(np.allclose(x, y, rtol=1e-3) for x, y in zip(result.params, ref))


def test_average_gate_fidelity_qutip_comparison() -> None:
    # Parameters from test_two_photon_cz
    params = (
        7.600019896010689,
        [-49.92218101, 50],
        [1.75873066, -0.61830304],
        [10, 10],
    )
    duration = params[0]
    phase_params_jnp = jnp.array(params[2])

    # --- Rydopt average gate fidelity (adiabatic elimination) ---
    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0)
    lower = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.sin_crab,
        rabi_ansatz=ro.pulses.const,
    )
    upper = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        rabi_ansatz=ro.pulses.const,
    )
    pulse = ro.pulses.TwoPhotonPulseAnsatz(
        lower_transition=lower,
        upper_transition=upper,
        lower_param_counts=(1, 2, 1),
        decay=0,
    )
    rydopt_fid = float(ro.simulation.average_gate_fidelity(gate, pulse, params))

    # --- QuTiP average gate fidelity with full 3-level atoms ---
    def qutip_avg_gate_fidelity(Delta_l: float, Delta_u: float, Omega_l: float, Omega_u: float) -> float:
        tlist = np.linspace(0, duration, 100)

        # Three-level basis: |1>, |e>, |r>
        ket1, kete, ketr = qt.basis(3, 0), qt.basis(3, 1), qt.basis(3, 2)
        I3 = qt.qeye(3)

        # Single-atom Hamiltonian
        t_grid = np.linspace(0, duration, 1000)
        xi_vals = np.array(ro.pulses.sin_crab(jnp.array(t_grid), duration, phase_params_jnp))
        xi_func = interp1d(t_grid, xi_vals, kind="cubic", fill_value="extrapolate")

        def coeff_m(t: float, args: None = None) -> complex:
            return np.exp(-1j * xi_func(float(t)))

        def coeff_p(t: float, args: None = None) -> complex:
            return np.exp(1j * xi_func(float(t)))

        H0_s = (
            -Delta_l * kete @ kete.dag()
            - (Delta_l + Delta_u) * ketr @ ketr.dag()
            + (Omega_u / 2) * (kete @ ketr.dag() + ketr @ kete.dag())
        )
        H_m_s = (Omega_l / 2) * ket1 @ kete.dag()
        H_p_s = (Omega_l / 2) * kete @ ket1.dag()

        # Single-atom evolution (|01> / |10> block)
        res_s = qt.sesolve([H0_s, [H_m_s, coeff_m], [H_p_s, coeff_p]], ket1, tlist)
        c_single = complex(res_s.states[-1].full()[0, 0])

        # Two-atom Hamiltonian (implement infinite interaction by projecting out |rr>)
        H0_2 = qt.tensor(H0_s, I3) + qt.tensor(I3, H0_s)
        Hm_2 = qt.tensor(H_m_s, I3) + qt.tensor(I3, H_m_s)
        Hp_2 = qt.tensor(H_p_s, I3) + qt.tensor(I3, H_p_s)

        P = qt.tensor(I3, I3) - qt.tensor(ketr, ketr) @ qt.tensor(ketr, ketr).dag()
        ix = list(range(8))
        H0_8 = qt.Qobj((P * H0_2 * P).full()[np.ix_(ix, ix)])
        Hm_8 = qt.Qobj((P * Hm_2 * P).full()[np.ix_(ix, ix)])
        Hp_8 = qt.Qobj((P * Hp_2 * P).full()[np.ix_(ix, ix)])

        # Two-atom evolution (|11> block) with V_nn = inf
        res_2 = qt.sesolve(
            [H0_8, [Hm_8, coeff_m], [Hp_8, coeff_p]],
            qt.basis(8, 0),
            tlist,
        )
        c_two = complex(res_2.states[-1].full()[0, 0])

        # Average gate fidelity using QuTiP (no manual F_pro -> F_avg formula)
        U_eff = qt.Qobj(np.diag([1.0, c_single, c_single, c_two]))
        phi = np.angle(c_single)
        U_tgt = qt.Qobj(np.diag([1.0, np.exp(1j * phi), np.exp(1j * phi), np.exp(1j * (2 * phi + np.pi))]))
        U_err = U_tgt.dag() @ U_eff
        return float(qt.average_gate_fidelity(qt.to_super(U_err)))

    # --- Compare at original parameters ---
    qutip_fid = qutip_avg_gate_fidelity(params[1][0], params[1][1], params[3][0], params[3][1])
    diff = abs(rydopt_fid - qutip_fid)

    assert rydopt_fid > 1 - 1e-7
    assert qutip_fid > 0.98
    assert diff < 0.02

    # --- Verify the difference is due to adiabatic elimination ---
    # Scale the intermediate-state detuning by alpha while keeping the effective
    # two-level Hamiltonian parameters constant.
    # Adiabatic elimination becomes exact as alpha to infinity, so the two fidelities
    # must converge.
    alpha = 5.0
    Delta_l_orig = params[1][0]
    Delta_u_orig = params[1][1]
    Omega_orig = params[3][0]

    Delta_l_sc = alpha * Delta_l_orig
    Delta_eff = Delta_l_orig + Delta_u_orig  # no ac-Stark shift because omega_l=omega_u
    Delta_u_sc = Delta_eff - Delta_l_sc
    Omega_sc = np.sqrt(alpha) * Omega_orig

    # Rydopt fidelity is unchanged (same effective Hamiltonian)
    params_sc = (duration, [Delta_l_sc, Delta_u_sc], list(params[2]), [Omega_sc, Omega_sc])
    rydopt_fid_sc = float(ro.simulation.average_gate_fidelity(gate, pulse, params_sc))
    assert abs(rydopt_fid_sc - rydopt_fid) < 1e-7

    # Qutip fidelity converges toward rydopt as adiabatic elimination improves
    qutip_fid_sc = qutip_avg_gate_fidelity(Delta_l_sc, Delta_u_sc, Omega_sc, Omega_sc)
    diff_sc = abs(rydopt_fid_sc - qutip_fid_sc)

    assert diff_sc < diff
    assert diff_sc < 0.005
