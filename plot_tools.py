"""Plotting tools for results."""
from typing import Optional, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from .common_types import Plot, Table
from .processing_tools import combine_columns
from .spectrum import Spectrum

def get_freq_lim(freq_range: Tuple[u.Quantity, u.Quantity],
                 ratio: float = 1/2) -> Tuple[float, float]:
    """Calculate the frequency limits from range.

    Args:
      freq_range: frequency range
      ratio: optional; fraction of the range width to use as limit border.
    """
    width = np.abs(freq_range[0] - freq_range[1])
    border = width.value * ratio
    return freq_range[0].value - border, freq_range[1].value + border

def plot_markers(ax: plt.Axes, table: Table, key: str,
                 color_key: Optional[str] = None,
                 top: Optional[int] = None,
                 plot_null: bool = False) -> None:
    """Plot frequency marker with values from table.

    Args:
      ax: plot axis.
      table: table with values.
      key: table column to plot.
      color_key: optional; column to use for colors.
      top: optional; only plot the top n values.
      plot_null: optional; plot invalid values?
    """
    # Defaults
    labels = {'log10Aij': r'$\log A_{ij}$'}
    #colors = ['#330000', '#550000', '#770000', '#990000', '#bb0000', '#dd0000',
    #          '#ff0000', '#ff2222', '#ff4444', '#ff6666', '#ff8888']

    # Checks
    if color_key is None:
        color_key = key

    # Plot markers
    freqs = combine_columns(table, ['Freq', 'MeasFreq'])
    vals = table[key]
    color = table[color_key]
    leg_labels = table['Species']
    if not plot_null:
        ind = np.flatnonzero(table[key])
        freqs = freqs[ind]
        vals = vals[ind]
        color = color[ind]
        leg_labels = leg_labels[ind]
    if top:
        freqs = freqs[:top]
        vals = vals[:top]
        color = color[:top]
        leg_labels = leg_labels[:top]
    scatter = ax.scatter(freqs, vals, marker=2, cmap='viridis', s=100, c=color,
                         vmin=np.nanmin(color), vmax=np.nanmax(color))

    # Label
    ax.set_ylabel(labels.get(key), color='r')
    handles, *_ = scatter.legend_elements(prop='colors', num=list(color),
            size=15)
    ax.legend(handles, leg_labels, loc=(1.2, 0.0))

def plot_spectrum(spec: Spectrum,
                  freq_range: Tuple[u.Quantity],
                  table: Optional[Table] = None,
                  key: Optional[str] = None,
                  color_key: Optional[str] = None,
                  top: Optional[int] = None) -> Plot:
    """Plot spectrum.

    Args:
      spec: spectrum.
      freq_range: frequency range.
      table: optiona; table with values for markers.
      key: optiona; table column to extract markers.
      color_key: optional; column to use for marker colors.
      top: optional; only plot the top n values from table.
    """
    # Figure
    plt.close()
    fig = plt.figure(figsize=(5, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    # Plot spectrum
    ax1.plot(spec.spectral_axis, spec.intensity, 'b-')
    xlim = get_freq_lim(freq_range)
    ax1.set_xlim(*xlim)
    cent = spec.centroid(*freq_range)
    ax1.axvline(cent.value, ls='--', c='#6f6f6f')
    ax1.set_ylim(bottom=-0.001)
    ax1.set_ylabel(f'Intensity ({spec.intensity.unit:latex_inline})',
                   color='b')
    ax1.set_xlabel(f'Frequency ({spec.spectral_axis.unit:latex_inline})')

    # Spans
    ax1.axvspan(xlim[0], freq_range[0].value, facecolor='#6f6f6f', alpha=0.5)
    ax1.axvspan(freq_range[1].value, xlim[1], facecolor='#6f6f6f', alpha=0.5)

    # Plot table key
    if table is not None and key:
        plot_markers(ax2, table, key, top=top, color_key=color_key)

    return fig, ax1, ax2

