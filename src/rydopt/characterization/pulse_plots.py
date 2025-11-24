from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import ParamsTuple


def plot_pulse(
    pulse_ansatz: PulseAnsatz,
    params: ParamsTuple,
    plot_detuning: bool = True,
    plot_phase: bool = True,
    plot_rabi: bool = True,
    phase_offset: bool = False,
):
    r"""Function that plots a pulse, given the pulse ansatz and the pulse parameters.

    Example:
        >>> import rydopt as ro
        >>> pulse_ansatz = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab
        ... )
        >>> params = (7.61140652, (-0.07842706,), (1.80300902, -0.61792703), ())
        >>> plot_pulse(pulse_ansatz, params)

    Args:
        pulse_ansatz: ansatz of the gate pulse.
        params: pulse parameters.
        plot_detuning: whether to plot the detuning pulse.
        plot_phase: whether to plot the phase pulse.
        plot_rabi: whether to plot the rabi pulse.
        phase_offset: let the phase pulse begin at 0.
    """
    T = params[0]
    ts = np.linspace(0, T, 1000)
    detuning_pulse, phase_pulse, rabi_pulse = pulse_ansatz.make_pulses(params)

    if phase_offset:
        offset = phase_pulse(0)
    else:
        offset = 0

    plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(layout="constrained", figsize=(3.4, 1.9))
    fig.set_dpi(300)
    y_label_string = r""
    if plot_rabi:
        ax.plot(
            ts,
            rabi_pulse(ts),
            linewidth=2,
            color="tab:gray",
            linestyle="dotted",
            label=r"Rabi freq. $\Omega$",
        )
        y_label_string += r"$\Omega/\Omega_0 \quad$"
    if plot_detuning:
        ax.plot(
            ts,
            detuning_pulse(ts),
            linewidth=2,
            color="tab:blue",
            linestyle="dashed",
            label=r"detuning $\Delta$",
        )
        y_label_string += r"$\Delta/\Omega_0 \quad$"
    if plot_phase:
        ax.plot(
            ts,
            phase_pulse(ts) - offset,
            linewidth=2,
            color="tab:orange",
            linestyle="solid",
            label=r"phase $\xi$",
        )
        y_label_string += r"$\xi$"
    ax.set_xlabel(r"$\Omega_0 t$", fontsize=10)
    ax.set_ylabel(y_label_string, fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=8)
    ax.grid()
    plt.show()
