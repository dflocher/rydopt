import numpy as np
import matplotlib.pyplot as plt
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import FloatParams


def _plot_subpulses(
    pulse_ansatz, params, phase_offset, plot_detuning, plot_phase, plot_rabi
):
    T, detuning_params, phase_params, rabi_params = params
    detuning_params = np.asarray(detuning_params)
    phase_params = np.asarray(phase_params)
    rabi_params = np.asarray(rabi_params)

    ts = np.linspace(0, T, 1000)
    detuning_pulse = pulse_ansatz.detuning_ansatz(ts, T, detuning_params)  # noqa: E731
    phase_pulse = pulse_ansatz.phase_ansatz(ts, T, phase_params)  # noqa: E731
    rabi_pulse = pulse_ansatz.rabi_ansatz(ts, T, rabi_params)  # noqa: E731

    if phase_offset:
        phase_pulse -= phase_pulse[0]

    plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(layout="constrained", figsize=(3.4, 1.9))
    fig.set_dpi(300)
    y_label_string = r""
    if plot_rabi:
        ax.plot(
            ts,
            rabi_pulse,
            linewidth=2,
            color="tab:gray",
            linestyle="dotted",
            label=r"Rabi freq. $\Omega$",
        )
        y_label_string += r"$\Omega/\Omega_0 \quad$"
    if plot_detuning:
        ax.plot(
            ts,
            detuning_pulse,
            linewidth=2,
            color="tab:blue",
            linestyle="dashed",
            label=r"detuning $\Delta$",
        )
        y_label_string += r"$\Delta/\Omega_0 \quad$"
    if plot_phase:
        ax.plot(
            ts,
            phase_pulse,
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


def plot_pulse(
    pulse_ansatz: PulseAnsatz,
    params: tuple[FloatParams, ...],
    phase_offset=False,
):
    _plot_subpulses(
        pulse_ansatz,
        params,
        phase_offset,
        plot_detuning=True,
        plot_phase=True,
        plot_rabi=True,
    )


def plot_pulse_without_defaults(
    pulse_ansatz: PulseAnsatz,
    params: tuple[FloatParams, ...],
    phase_offset=False,
):
    T, detuning_params, phase_params, rabi_params = params
    _plot_subpulses(
        pulse_ansatz,
        params,
        phase_offset,
        len(detuning_params) > 0,
        len(phase_params) > 0,
        len(rabi_params) > 0,
    )
