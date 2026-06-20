"""Plotting helpers for the ptychography training subpackage."""

from __future__ import annotations

import numpy as np

__all__ = ["plot_scattering_power", "plot_training_pair"]

# First atomic number of each period (H, Li, Na, K, Rb, Cs, Fr).
_PERIOD_STARTS = (1, 3, 11, 19, 37, 55, 87)


def plot_scattering_power(
    parametrization: str = "lobato",
    *,
    z_max: int = 92,
    highlight=None,
    ax=None,
):
    """Plot per-element scattering power vs atomic number.

    Scattering power is the integrated projected potential of a single
    atom (:func:`tricor.ptycho.potential.scattering_power`) — the weight
    each species carries in the (blurred) ptychographic potential.  The
    periodic-table rows are shaded as coloured background bands.

    Parameters
    ----------
    parametrization
        abTEM potential parametrization the table was built with.
    z_max
        Largest atomic number to plot.
    highlight
        Atomic number(s) to mark (e.g. the species present in a
        structure: ``atoms.numbers``).
    ax
        Existing Axes; a wide one is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from .potential import scattering_power

    z = np.arange(1, z_max + 1)
    s = scattering_power(z, parametrization=parametrization)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3.2))

    # Period background bands.
    starts = list(_PERIOD_STARTS)
    bounds = starts + [z_max + 1]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, lo in enumerate(starts):
        hi = bounds[i + 1] - 1
        if lo > z_max:
            break
        ax.axvspan(lo - 0.5, min(hi, z_max) + 0.5, color=colors[i], alpha=0.12, lw=0)
        ax.text(
            (lo + min(hi, z_max)) / 2, 0.97, f"period {i + 1}",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color="0.4",
        )

    ax.plot(z, s, "-o", ms=3, lw=1.0, color="#222")

    if highlight is not None:
        hz = np.unique(np.atleast_1d(np.asarray(highlight, dtype=int)))
        hz = hz[(hz >= 1) & (hz <= z_max)]
        if hz.size:
            ax.plot(hz, scattering_power(hz, parametrization=parametrization),
                    "o", ms=9, color="#e02424", zorder=5, label="present")
            ax.legend(loc="upper left", fontsize=9)

    ax.set_xlim(0.5, z_max + 0.5)
    ax.set_xlabel("atomic number Z")
    ax.set_ylabel("scattering power (eV·Å³)")
    ax.set_title(f"Integrated projected potential per element ({parametrization})")
    ax.figure.tight_layout()
    return ax


def plot_training_pair(pairs, index: int):
    """Plot one training pair: input window, g2, and g3 slice.

    Parameters
    ----------
    pairs
        List from :func:`tricor.ptycho.sliding_window_pairs`.
    index
        Which pair to show.

    Returns
    -------
    numpy.ndarray of matplotlib Axes (length 3).
    """
    import matplotlib.pyplot as plt

    from .._plotting import show_2d

    p = pairs[index]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))

    show_2d(
        p["input"].T,
        ax=ax[0],
        cmap="gray",
        title=f"input window  (cx={p['cx']:.0f}, cy={p['cy']:.0f}, z0={p['z0']:.0f} Å)",
        xlabel="x (px)",
        ylabel="y (px)",
        colorbar=True,
    )
    ax[1].plot(p["r"], p["g2"])
    ax[1].axhline(1, ls="--", c="gray")
    ax[1].set(xlabel="r (Å)", ylabel="weighted g2", title="g2")
    show_2d(
        p["g3_slice"],
        extent=[0, p["r"][-1], 0, 180],
        ax=ax[2],
        cmap="RdBu_r",
        aspect="auto",
        vmin=0,
        vmax=2,
        title="g3 slice (r01 ~ NN)",
        xlabel="r02 (Å)",
        ylabel="angle (deg)",
    )
    fig.tight_layout()
    return ax
