from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey

from rydopt.pulses.pulse_ansatz import PulseAnsatz


def plot_spectrum(
    pulse_ansatz: PulseAnsatz,
    params,
    *,
    num_points: int = 512,
    pad_factor: int = 128,
    tapered: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Function that plots the spectrum of a pulse, given the pulse ansatz and the pulse parameters.

    Args:
        pulse_ansatz: Ansatz of the gate pulse.
        params: Pulse parameters.
        num_points: Number of sampling points in the physical time interval.
        pad_factor: Factor by which the time array is padded.
        tapered: If True, applies a Tukey window in the padded region.
        xlim: Optional x-axis (frequency) limits; if None, chosen automatically.
        ylim: Optional y-axis (dB) limits; if None, chosen automatically.
        ax: Optional matplotlib Axes to draw on; if None, a new one is created.

    Returns:
        A tuple of (Figure, Axes) containing the spectrum plot.

    """
    duration = params[0]

    # Padded times
    times = jnp.linspace(
        -duration * (pad_factor - 1) / 2, duration * (pad_factor + 1) / 2, num_points * pad_factor, endpoint=False
    )

    # Evaluated pulses
    pulses = pulse_ansatz.evaluate_pulse_functions(times, params)
    labels = [
        r"$\mathcal{F}\left(\Delta(t \Omega_0)\right)$",
        r"$\mathcal{F}\left(\phi(t \Omega_0)\right)$",
        r"$\mathcal{F}\left(\Omega(t \Omega_0)\right)$",
    ]
    is_constant = [np.all(p == p[0]) for p in pulses]

    # Tukey window: flat on the physical interval, tapered only in the padded region
    win = tukey(len(times), alpha=(pad_factor - 1) / pad_factor) if tapered else 1.0

    # Calculate spectra
    freqs = np.fft.rfftfreq(len(times), d=times[1] - times[0])
    spectra = [np.abs(np.fft.rfft(p * win)) for p in pulses]

    # Convert spectra to Decibel
    eps = np.finfo(float).tiny
    spectra = [20.0 * np.log10(np.maximum(s / np.maximum(np.max(s), eps), eps)) for s in spectra]

    # Plot spectra
    owns_ax = ax is None

    if owns_ax:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    else:
        assert ax is not None
        fig = cast(plt.Figure, ax.figure)

    if owns_ax and ylim is None:
        ylim = (-80, 5)

    if owns_ax and xlim is None:
        assert ylim is not None
        max_spectra = np.max(np.vstack(spectra), axis=0)
        indices = np.where(max_spectra >= ylim[0])[0]
        last_idx = indices.max() if indices.size > 0 else len(freqs) - 1
        idx = min(last_idx + 1, len(freqs) - 1)
        xlim = (0.0, freqs[idx])

    for spectrum, label, skip in zip(spectra, labels, is_constant):
        if skip:
            continue
        if ylim is not None and np.all(spectrum[1:] < ylim[0]):
            continue
        ax.plot(freqs, spectrum, label=label)

    if owns_ax:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$f / \Omega_0$")
        ax.set_ylabel("Amplitude (dB)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

    return fig, ax
