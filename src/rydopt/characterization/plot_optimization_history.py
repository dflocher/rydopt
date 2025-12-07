from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from rydopt.optimization import OptimizationResult


def plot_optimization_history(
    optimization_result: OptimizationResult,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Function that plots the optimization history.

    Args:
        optimization_result: OptimizationResult object.
        xlim: Optional x-axis (optimization steps) limits; if None, chosen automatically.
        ylim: Optional y-axis (infidelity) limits; if None, chosen automatically.
        ax: Optional matplotlib Axes to draw on; if None, a new one is created.

    Returns:
        A tuple of (Figure, Axes) containing the optimization history plot.

    """
    tol = optimization_result.tol
    history = optimization_result.history

    # Plot history
    owns_ax = ax is None

    if owns_ax:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=160)
    else:
        assert ax is not None
        fig = cast(plt.Figure, ax.figure)

    if owns_ax and ylim is None:
        ylim = (tol, 5)

    if owns_ax and xlim is None:
        assert ylim is not None
        max_infidelity = np.max(history, axis=1)
        indices = np.where(max_infidelity >= ylim[0])[0]
        last_idx = indices.max() if indices.size > 0 else len(history) - 1
        idx = min(last_idx + 1, len(history) - 1)
        xlim = (0, idx)

    ax.plot(history)

    if owns_ax:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xmargin(0)
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Infidelity")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
        fig.tight_layout()

    return fig, ax
