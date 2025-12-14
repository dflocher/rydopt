from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey

from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.types import PulseParams


def plot_spectrum(
    pulse: PulseAnsatz,
    params: PulseParams,
    *,
    plot_detuning: bool = True,
    plot_phase: bool = True,
    plot_rabi: bool = True,
    num_points: int = 256,
    pad_factor: int = 1024,
    tapered: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Function that plots the spectrum of a pulse, given the pulse ansatz and the pulse parameters.

    Example:
        >>> import rydopt as ro
        >>> pulse = ro.pulses.PulseAnsatz(
        ...     detuning_ansatz=ro.pulses.const,
        ...     phase_ansatz=ro.pulses.sin_crab,
        ... )
        >>> params = (7.6, [-0.1], [1.8, -0.6], [])
        >>> ro.characterization.plot_spectrum(pulse, params)
        (<Figure ...

    Args:
        pulse: Ansatz of the gate pulse.
        params: Pulse parameters.
        plot_detuning: Whether to plot the detuning pulse, default is True.
        plot_phase: Whether to plot the phase pulse, default is True.
        plot_rabi: Whether to plot the rabi pulse, default is True.
        num_points: Number of sampling points in the time interval.
        pad_factor: Factor by which the time array is padded.
        tapered: If True, applies a Tukey window in the padded region.
        xlim: Optional x-axis (frequency) limits; if None, chosen automatically.
        ylim: Optional y-axis (dB) limits; if None, chosen automatically.
        ax: Optional :class:`matplotlib.axes.Axes` to draw on; if None, a new one is created.

    Returns:
        A tuple of (fig, ax) where ax is the axes used for the spectrum plot.

    """
    duration = params[0]

    times = jnp.linspace(
        -duration * (pad_factor - 1) / 2, duration * (pad_factor + 1) / 2, num_points * pad_factor, endpoint=False
    )

    # Evaluated pulse
    selector = [plot_detuning, plot_phase, plot_rabi]
    values = np.array(pulse.evaluate_pulse_functions(times, params))[selector]
    labels = np.array(
        [
            r"$\mathcal{F}\left(\Delta(t)\right)$",
            r"$\mathcal{F}\left(\xi(t)\right)$",
            r"$\mathcal{F}\left(\Omega(t)\right)$",
        ]
    )[selector]
    is_constant = [np.all(v == v[0]) for v in values]

    # Tukey window: flat on the physical interval, tapered only in the padded region
    win = tukey(len(times), alpha=(pad_factor - 1) / pad_factor) if tapered else 1.0

    # Calculate spectra
    freqs = np.fft.rfftfreq(len(times), d=times[1] - times[0])
    spectra = [np.abs(np.fft.rfft(v * win)) for v in values]

    # Convert spectra to Decibel
    eps = np.finfo(float).tiny
    spectra = [20.0 * np.log10(np.maximum(s / np.maximum(np.max(s), eps), eps)) for s in spectra]

    # Plot spectra
    owns_ax = ax is None

    if owns_ax:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=160)
    else:
        assert ax is not None
        fig = cast(plt.Figure, ax.figure)

    if owns_ax and ylim is None:
        ylim = (-100, 5)

    if owns_ax and xlim is None:
        assert ylim is not None
        max_spectra = np.max(np.vstack(spectra), axis=0)
        indices = np.where(max_spectra >= ylim[0])[0]
        last_idx = indices.max() if indices.size > 0 else len(freqs) - 1
        idx = min(last_idx + 1, len(freqs) - 1)
        xlim = (0.0, freqs[idx])

    for spectrum, label, skip in zip(spectra, labels, is_constant):
        if skip:
            if owns_ax:
                ax.plot([], [])  # propagate the color cycler
            continue
        if ylim is not None and np.all(spectrum[1:] < ylim[0]):
            if owns_ax:
                ax.plot([], [])  # propagate the color cycler
            continue
        ax.plot(freqs, spectrum, label=label)

    if owns_ax:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$f / \Omega_0$")
        ax.set_ylabel("Amplitude [dB]")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

    return fig, ax
