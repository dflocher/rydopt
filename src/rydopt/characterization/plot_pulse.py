from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import PulseParams


def plot_pulse(
    pulse: PulseAnsatz,
    params: PulseParams,
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
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> params = (7.6, [-0.1], [1.8, -0.6], [])
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

    # Evaluated pulse
    selector = [plot_detuning, plot_phase, plot_rabi]

    values = np.array(pulse.evaluate_pulse_functions(times, params))
    if subtract_phase_offset:
        values[1] -= values[1][0]
    values = values[selector]

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
