from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rydopt.gates import GateFamily
from rydopt.protocols import PulseAnsatzLike
from rydopt.pulses import PulseFamilyAnsatz
from rydopt.types import ParamsFloatLike


def _evaluate_pulse(
    pulse: PulseAnsatzLike,
    params: ParamsFloatLike,
    times: jnp.ndarray,
    *,
    plot_detuning: bool,
    plot_phase: bool,
    plot_rabi: bool,
    subtract_phase_offset: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Evaluate pulse functions and return data ready for plotting."""
    selector = [plot_detuning, plot_phase, plot_rabi]

    values = np.array(pulse.evaluate_pulse_functions(times, params))
    values[1] -= values[0]
    values = values[1:][selector]

    if subtract_phase_offset:
        values[1] -= values[1][0]

    labels = np.array(
        [
            r"$\Delta(t)$",
            r"$\xi(t)$",
            r"$\Omega(t)$",
        ]
    )[selector]

    ylabel = ", ".join(
        np.array(
            [
                r"$\Delta / \Omega_0$",
                r"$\xi$ [rad]",
                r"$\Omega / \Omega_0$",
            ]
        )[selector]
    )

    return values, labels, ylabel


def plot_pulse(
    pulse: PulseAnsatzLike,
    params: ParamsFloatLike,
    *,
    plot_detuning: bool = True,
    plot_phase: bool = True,
    plot_rabi: bool = True,
    subtract_phase_offset: bool = False,
    num_points: int = 1024,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Function that plots a pulse, given the pulse ansatz and the pulse parameters.

    Example:
        >>> import rydopt as ro
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.Const(),
        ...     phase_ansatz=ro.pulses.SinCrab(2),
        ... )
        >>> params = ro.pulses.PulseParams(7.6, [-0.1], [1.8, -0.6], [])
        >>> ro.characterization.plot_pulse(pulse, params)
        (<Figure ...

    Args:
        pulse: Ansatz of the gate pulse.
        params: Pulse parameters.
        plot_detuning: Whether to plot the detuning pulse, default is True.
        plot_phase: Whether to plot the phase pulse, default is True.
        plot_rabi: Whether to plot the rabi pulse, default is True.
        subtract_phase_offset: Whether the phase pulse begins at 0, default is False.
        num_points: Number of sampling points in the time interval.
        ax: Optional :class:`matplotlib.axes.Axes` to draw on; if None, a new one is created.

    Returns:
        A tuple of (fig, ax) where ax is the axes used for the pulse plot.

    """
    duration = params[0]
    times = jnp.linspace(0, duration, num_points)

    values, labels, ylabel = _evaluate_pulse(
        pulse,
        params,
        times,
        plot_detuning=plot_detuning,
        plot_phase=plot_phase,
        plot_rabi=plot_rabi,
        subtract_phase_offset=subtract_phase_offset,
    )

    owns_ax = ax is None

    if owns_ax:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=160)
    else:
        assert ax is not None
        fig = cast(plt.Figure, ax.figure)

    for v, label in zip(values, labels):
        ax.plot(times, v, label=label)

    if owns_ax:
        ax.set_xmargin(0)
        ax.set_xlabel(r"$t \Omega_0$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

    return fig, ax


def plot_pulse_family(
    pulse_family: PulseFamilyAnsatz,
    family_params: ParamsFloatLike,
    gate_family: GateFamily,
    *,
    plot_detuning: bool = True,
    plot_phase: bool = True,
    plot_rabi: bool = True,
    subtract_phase_offset: bool = False,
    num_points: int = 1024,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Function that plots a set of pulses, given the pulse family ansatz, the pulse family
    parameters and the gate family.

    Args:
        pulse_family: Ansatz of the pulse family
        family_params: Pulse family parameters.
        gate_family: an instance of the GateFamily
        plot_detuning: Whether to plot the detuning pulse, default is True.
        plot_phase: Whether to plot the phase pulse, default is True.
        plot_rabi: Whether to plot the rabi pulse, default is True.
        subtract_phase_offset: Whether the phase pulse begins at 0, default is False.
        num_points: Number of sampling points in the time interval.
        ax: Optional :class:`matplotlib.axes.Axes` to draw on; if None, a new one is created.

    Returns:
        A tuple of (fig, ax) where ax is the axes used for the pulse plot.

    """
    owns_ax = ax is None

    if owns_ax:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=160)
    else:
        assert ax is not None
        fig = cast(plt.Figure, ax.figure)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", ":"]
    ylabel = ""

    for i, gate_param in enumerate(gate_family.parameter_values):
        pulse = pulse_family.generate_pulse_ansatz(gate_param)
        params = pulse_family.generate_pulse_params(family_params, gate_param)

        duration = params[0]
        times = jnp.linspace(0, duration, num_points)

        values, _, ylabel = _evaluate_pulse(
            pulse,
            params,
            times,
            plot_detuning=plot_detuning,
            plot_phase=plot_phase,
            plot_rabi=plot_rabi,
            subtract_phase_offset=subtract_phase_offset,
        )

        label = rf"${gate_param / np.pi:.2f}\,\pi$"

        for count, v in enumerate(values):
            ax.plot(
                times,
                v,
                label=label if count == 0 else None,
                color=colors[i],
                linestyle=linestyles[count],
            )

    if owns_ax:
        ax.set_xmargin(0)
        ax.set_xlabel(r"$t \Omega_0$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

    return fig, ax
