from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from rydopt.pulses.general_pulse_ansatz_functions import const
from rydopt.types import PulseAnsatzFunction, PulseParams


def _const_zero(t: jax.Array | float, _duration: float, _ansatz_params: jax.Array) -> jax.Array:
    return const(t, _duration, jnp.array([0.0]))


def _const_one(t: jax.Array | float, _duration: float, _ansatz_params: jax.Array) -> jax.Array:
    return const(t, _duration, jnp.array([1.0]))


@dataclass
class PulseAnsatz:
    r"""Data class that stores ansatz functions for the laser pulse that couples the qubit state :math:`|1\rangle` to
    the Rydberg state :math:`|r\rangle`.

    RydOpt models the atom-light interaction in the rotating frame, using the rotating wave approximation. The
    Hamiltonian of the driven two-level ladder system :math:`|1\rangle \leftrightarrow |r\rangle`
    is

    .. math::

            H_\mathrm{drive}(t)=\begin{pmatrix}
                0 & \frac{\Omega(t)}{2} e^{-i\xi(t)} \\
                \frac{\Omega(t)}{2} e^{i\xi(t)} & -\Delta(t)
            \end{pmatrix}.

    For available ansatz functions for the detuning :math:`\Delta(t)`, phase :math:`\xi(t)`, and Rabi
    frequency :math:`\Omega(t)` sweeps, see below.
    The function :func:`optimize <rydopt.optimization.optimize>` allows optimizing the
    parameters of the ansatz functions and duration of the laser pulse
    to maximize the gate fidelity. Initial parameters can be provided to the function
    as :class:`PulseParams`, i.e., as a tuple ``(duration, detuning_params, phase_params, rabi_params)``.

    Example:
        >>> import rydopt as ro
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )

    Attributes:
        detuning_ansatz: Detuning sweep :math:`\Delta(t)`, default is zero.
        phase_ansatz: Phase sweep :math:`\xi(t)`, default is zero.
        rabi_ansatz: Rabi frequency amplitude sweep :math:`\Omega(t)`, default is one.

    """

    detuning_ansatz: PulseAnsatzFunction = _const_zero
    phase_ansatz: PulseAnsatzFunction = _const_zero
    rabi_ansatz: PulseAnsatzFunction = _const_one

    def evaluate_pulse_functions(
        self, t: jax.Array | float, params: PulseParams
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Evaluate the detuning, phase, and the rabi sweeps for fixed
        parameters at the given times.

        Args:
            t: Time samples at which the functions are evaluated
            params: Pulse parameters

        Returns:
            Tuple ``(detuning_1, detuning_r, phase, rabi)``

        """
        duration, detuning_ansatz_params, phase_ansatz_params, rabi_ansatz_params = params
        detuning_ansatz_params = jnp.asarray(detuning_ansatz_params)
        phase_ansatz_params = jnp.asarray(phase_ansatz_params)
        rabi_ansatz_params = jnp.asarray(rabi_ansatz_params)

        return (
            jnp.zeros_like(t),
            self.detuning_ansatz(t, duration, detuning_ansatz_params),
            self.phase_ansatz(t, duration, phase_ansatz_params),
            self.rabi_ansatz(t, duration, rabi_ansatz_params),
        )


@dataclass
class TwoPhotonPulseAnsatz:
    r"""Data class that stores an effective two-photon pulse ansatz that couples the qubit
    state :math:`|1\rangle` to the Rydberg state :math:`|r\rangle` via the intermediate state :math:`|e\rangle`.

    RydOpt models the atom-light interaction in the rotating frame, using the rotating wave approximation. The
    Hamiltonian of the driven three-level ladder system
    :math:`|1\rangle \leftrightarrow |e\rangle \leftrightarrow |r\rangle` is taken as

    .. math::

        H_\mathrm{3lvl}(t)=
        \begin{pmatrix}
            0 &
            \frac{\Omega_\ell(t)}{2}\,e^{-i\xi_\ell(t)} &
            0 \\[6pt]
            \frac{\Omega_\ell(t)}{2}\,e^{i\xi_\ell(t)} &
            -\Delta_\ell(t) - i \frac{\gamma}{2}&
            \frac{\Omega_u(t)}{2}\,e^{-i\xi_u(t)} \\[6pt]
            0 &
            \frac{\Omega_u(t)}{2}\,e^{i\xi_u(t)} &
            -\Delta_\ell(t)-\Delta_u(t)
        \end{pmatrix},

    where the lower/upper laser couples :math:`|1\rangle \leftrightarrow |e\rangle` /
    :math:`|e\rangle \leftrightarrow |r\rangle`
    with Rabi frequency amplitudes :math:`\Omega_{\ell/u}(t)`, phases :math:`\xi_{\ell/u}(t)`,
    detunings :math:`\Delta_{\ell/u}(t)`. :math:`\gamma` is the decay rate of the intermediate state.

    The implementation is restricted to the adiabatic-elimination regime
    (:math:`|\Delta_\ell| \gg |\Omega_\ell|, |\Omega_u|, |\delta|`
    and :math:`|\Delta_\ell|^2 \gg |\dot{\Omega}_\ell|, |\dot{\Omega}_u|, |\dot{\delta}|`
    with :math:`\delta = \Delta_\ell+\Delta_u`), where the system can be treated
    by an effective two-level Hamiltonian on the subspace :math:`\{|1\rangle,|r\rangle\}`:

    .. math::

        H_\mathrm{drive}(t)=
        \begin{pmatrix}
            -\Delta_{1,\mathrm{eff}}(t) & \frac{\Omega_\mathrm{eff}(t)}{2} e^{-i\xi_\mathrm{eff}(t)} \\
            \frac{\Omega_\mathrm{eff}(t)}{2} e^{i\xi_\mathrm{eff}(t)} & -\Delta_{r,\mathrm{eff}}(t)
        \end{pmatrix}.

    The effective controls are computed as

    .. math::

        \Omega_\mathrm{eff}(t)&=\frac{\Omega_\ell(t)\Omega_u(t)}{2(\Delta_\ell(t)+i\gamma/2)}, \\
        \xi_\mathrm{eff}(t)&=\xi_\ell(t)+\xi_u(t), \\
        \Delta_{1,\mathrm{eff}}(t)&=-
        \frac{\Omega_\ell(t)^2}{4(\Delta_\ell(t)+i\gamma/2)} \\
        \Delta_{r,\mathrm{eff}}(t)&=\Delta_\ell(t)+\Delta_u(t)-
        \frac{\Omega_u(t)^2}{4(\Delta_\ell(t)+i\gamma/2)}.

    For available ansatz functions for the detuning, phase, and Rabi frequency sweeps, see below.
    The function :func:`optimize <rydopt.optimization.optimize>` allows optimizing the
    parameters of the ansatz functions and duration of the laser pulse
    to maximize the gate fidelity. Initial parameters can be provided to the function
    as :class:`PulseParams`, i.e., as a tuple ``(duration, detuning_params, phase_params, rabi_params)``.
    Each parameter array within the tuple is
    packed as ``[*lower_transition_params, *upper_transition_params]``. The split
    positions are set by ``lower_param_counts=(n_detuning, n_phase, n_rabi)``.

    Example:
        >>> import rydopt as ro
        >>> lower = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> upper = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     rabi_ansatz=ro.pulses.const,
        ... )
        >>> pulse = ro.pulses.TwoPhotonPulseAnsatz(
        ...     lower_transition=lower,
        ...     upper_transition=upper,
        ...     lower_param_counts=(1, 4, 0)
        ... )

    Attributes:
        lower_transition: Ansatz for the lower transition :math:`|1\rangle \leftrightarrow |e\rangle`.
        upper_transition: Ansatz for the upper transition :math:`|e\rangle \leftrightarrow |r\rangle`.
        lower_param_counts: Tuple ``(n_detuning, n_phase, n_rabi)`` specifying how many
            entries per parameter array belong to the lower transition.
        decay: Decay rate of the intermediate state, default is zero.

    """

    lower_transition: PulseAnsatz
    upper_transition: PulseAnsatz
    lower_param_counts: tuple[int, int, int]
    decay: float = 0.0

    @staticmethod
    def _split_1d(packed_params: ArrayLike, lower_count: int) -> tuple[jax.Array, jax.Array]:
        packed_params = jnp.asarray(packed_params)
        return packed_params[:lower_count], packed_params[lower_count:]

    def _unpack_transition_params(self, params: PulseParams) -> tuple[PulseParams, PulseParams]:
        duration, detuning_params, phase_params, rabi_params = params

        lower_detuning_count, lower_phase_count, lower_rabi_count = self.lower_param_counts

        lower_detuning_params, upper_detuning_params = self._split_1d(detuning_params, lower_detuning_count)
        lower_phase_params, upper_phase_params = self._split_1d(phase_params, lower_phase_count)
        lower_rabi_params, upper_rabi_params = self._split_1d(rabi_params, lower_rabi_count)

        lower_params: PulseParams = (duration, lower_detuning_params, lower_phase_params, lower_rabi_params)
        upper_params: PulseParams = (duration, upper_detuning_params, upper_phase_params, upper_rabi_params)
        return lower_params, upper_params

    def evaluate_pulse_functions(
        self, t: jax.Array | float, params: PulseParams
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Evaluate the effective two-photon detuning, phase, and the rabi sweeps for fixed
        parameters at the given times.

        Args:
            t: Time samples at which the functions are evaluated
            params: Pulse parameters

        Returns:
            Tuple ``(detuning_1, detuning_r, phase, rabi)``

        """
        lower_params, upper_params = self._unpack_transition_params(params)

        _, lower_detuning, lower_phase, lower_rabi = self.lower_transition.evaluate_pulse_functions(t, lower_params)
        _, upper_detuning, upper_phase, upper_rabi = self.upper_transition.evaluate_pulse_functions(t, upper_params)

        effective_rabi = lower_rabi * upper_rabi / (2.0 * (lower_detuning + 0.5j * self.decay))
        effective_phase = lower_phase + upper_phase
        effective_detuning_1 = - lower_rabi**2 / (4.0 * (lower_detuning + 0.5j * self.decay))
        effective_detuning_r = (lower_detuning+upper_detuning) - upper_rabi**2 / (4.0 * (lower_detuning + 0.5j * self.decay))

        return effective_detuning_1, effective_detuning_r, effective_phase, effective_rabi
