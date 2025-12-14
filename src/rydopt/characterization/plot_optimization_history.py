from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from rydopt.optimization import OptimizationResult


def plot_optimization_history(
    optimization_result: OptimizationResult,
    *,
    xlim_step: tuple[float, float] | None = None,
    xlim_duration: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax1: plt.Axes | None = None,
    ax2: plt.Axes | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes | None, plt.Axes | None]]:
    r"""Function that plots the optimization history.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> gate = ro.gates.TwoQubitGate(
        ...     phi=None,
        ...     theta=np.pi,
        ...     Vnn=float("inf"),
        ...     decay=0,
        ... )
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> initial_params = (7.6, [-0.1], [1.8, -0.6], [])
        >>> result = ro.optimization.optimize(
        ...     gate,
        ...     pulse,
        ...     initial_params,
        ...     num_steps=200,
        ...     tol=1e-7,
        ...     return_history=True,
        ... )
        Started optimization ...
        >>> plot_optimization_history(result)
        (<Figure ...

    Args:
        optimization_result: OptimizationResult object.
        xlim_step: Optional x-axis (optimization steps) limits; if None, chosen automatically.
        xlim_duration: Optional x-axis (gate duration) limits; if None, chosen automatically.
        ylim: Optional y-axis (infidelity) limits; if None, chosen automatically.
        ax1: Optional :class:`matplotlib.axes.Axes` to draw the infidelity as a function of the optimization step;
            if None and ax2 is also None, a new one is created.
        ax2: Optional :class:`matplotlib.axes.Axes` to draw the infidelity as a function of the gate duration;
            if None and ax1 is also None, a new one is created.

    Returns:
        A tuple (fig, (ax1, ax2)) where ax1 and ax2 are the axes used for the two plots.

    """
    tol = optimization_result.tol
    infidelity_history = optimization_result.infidelity_history
    duration_history = optimization_result.duration_history

    # Plot history
    owns_ax = ax1 is None and ax2 is None

    if owns_ax:
        fig, (ax1, ax2) = plt.subplots(figsize=(4, 3), dpi=160, nrows=2)
    elif ax1 is not None:
        fig = cast(plt.Figure, ax1.figure)
    else:
        assert ax2 is not None
        fig = cast(plt.Figure, ax2.figure)

    if owns_ax and ylim is None:
        ylim = (tol, 5)

    if owns_ax and xlim_step is None:
        assert ylim is not None
        max_infidelity = infidelity_history if infidelity_history.ndim == 1 else np.max(infidelity_history, axis=1)
        indices = np.where(max_infidelity >= ylim[0])[0]
        last_idx = indices.max() if indices.size > 0 else len(infidelity_history) - 1
        idx = min(last_idx + 1, len(infidelity_history) - 1)
        xlim_step = (0, idx)

    if owns_ax and xlim_duration is None:
        min_duration = np.min(duration_history)
        max_duration = np.max(duration_history)
        xlim_duration = (min_duration, max_duration)

    if ax1 is not None:
        ax1.plot(infidelity_history)

    if ax2 is not None:
        ax2.plot(duration_history, infidelity_history)

    if owns_ax:
        assert ax1 is not None
        ax1.set_xlim(xlim_step)
        ax1.set_ylim(ylim)
        ax1.set_xlabel("Optimization step")
        ax1.set_ylabel("$1-F$")
        ax1.grid(alpha=0.3)
        ax1.set_yscale("log")

        assert ax2 is not None
        ax2.set_xlim(xlim_duration)
        ax2.set_ylim(ylim)
        ax2.set_xlabel(r"Gate duration $T\Omega_0$")
        ax2.set_ylabel("$1-F$")
        ax2.grid(alpha=0.3)
        ax2.set_yscale("log")

        fig.tight_layout()

    return fig, (ax1, ax2)
