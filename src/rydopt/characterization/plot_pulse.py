from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import ParamsTuple


def plot_pulse(
    pulse_ansatz: PulseAnsatz,
    params: ParamsTuple,
    *,
    plot_detuning: bool = True,
    plot_phase: bool = True,
    plot_rabi: bool = True,
    subtract_phase_offset: bool = False,
    num_points: int = 1024,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Function that plots a pulse, given the pulse ansatz and the pulse parameters.

    Args:
        pulse_ansatz: Ansatz of the gate pulse.
        params: Pulse parameters.
        plot_detuning: Whether to plot the detuning pulse, default is True.
        plot_phase: Whether to plot the phase pulse, default is True.
        plot_rabi: Whether to plot the rabi pulse, default is True.
        subtract_phase_offset: Whether the phase pulse begins at 0, default is False.
        num_points: Number of sampling points in the time interval.
        ax: Optional matplotlib Axes to draw on; if None, a new one is created.

    Returns:
        A tuple of (Figure, Axes) containing the pulse plot.

    """
    duration = params[0]

    times = jnp.linspace(0, duration, num_points)

    # Evaluated pulse
    selector = [plot_detuning, plot_phase, plot_rabi]

    values = np.array(pulse_ansatz.evaluate_pulse_functions(times, params))
    if subtract_phase_offset:
        values[1] -= values[1][0]
    values = values[selector]

    labels = np.array(
        [
            r"$\Delta$",
            r"$\xi$",
            r"$\Omega$",
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

    # Plot pulse
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
