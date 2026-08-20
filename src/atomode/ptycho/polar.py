"""Rotation-invariant angular-symmetry features of a local image patch.

Reduce the neighbourhood around a window centre to a compact
``(n_channels, n_r)`` descriptor of its local angular symmetry.

The default pipeline (``mode="autocorrelation"``) is:

1. cut a circular Hann-windowed patch about the window centre,
2. zero-pad and form its **real-space autocorrelation**,
3. resample that onto a uniform polar grid ``(phi, r)``,
4. FFT along the angular axis and keep ``|F_m(r)|``,
5. taper the radial profile so it falls to zero at ``r_max``.

Why the autocorrelation
-----------------------
Decomposing the image *directly* about the window centre only reveals the
symmetry when the centre sits on a symmetry site: measured on an ideal
square lattice, the allowed/forbidden order contrast is ~1e11 exactly on a
site, 6.5x at 0.05 Å off it, and 0.6x (forbidden orders *exceeding*
allowed) at an arbitrary position.  No probe placement achieves that, so
the direct route is unusable in practice.

The autocorrelation is translation invariant by construction, so it gives
the correct signature wherever the window lands.  Verified on ideal
lattices sampled at arbitrary positions, with every disallowed channel at
or below 0.004:

===========  ==========================
lattice      channels present
===========  ==========================
square       4, 8, 12
triangular   6, 12
honeycomb    6, 12
===========  ==========================

It also wins on the real task.  Classifying atomode silicon by grain size
(five classes, chance 0.20) from pseudo-ptychographic potentials, at
several noise levels::

    descriptor          noise 0    noise 0.3    noise 1.0
    autocorrelation       0.62        0.63         0.71
    direct                0.23        0.21         0.27

Seven channels, not thirteen
----------------------------
An autocorrelation is centrosymmetric, ``A(-u) = A(u)``, so its angular
profile has period 180 deg and **all odd orders vanish identically**
(measured odd/even ~1e-14).  Storing them would waste six rows, so the
descriptor keeps the even orders only: ``0, 2, 4, 6, 8, 10, 12`` — seven
channels, all carrying signal.

The cost is that a 3-fold sp2 motif cannot be told from a 6-fold one by
the order pattern alone.  In practice that matters less than it sounds:
honeycomb and triangular lattices both show 6 and 12, but their *radial*
profiles differ strongly, and that separates them well.

``mode="direct"`` keeps the odd orders and returns all thirteen channels
``0 … 12``, at the cost of the probe-position sensitivity above.  It is
kept for comparison.
"""

from __future__ import annotations

from functools import lru_cache as _lru_cache

import numpy as np

__all__ = [
    "EVEN_ORDERS",
    "ALL_ORDERS",
    "default_orders",
    "polar_fft_features",
    "add_polar_features",
]

#: Even orders 0 … 12 — the seven channels an autocorrelation can carry.
EVEN_ORDERS = (0, 2, 4, 6, 8, 10, 12)
#: All orders 0 … 12 — thirteen channels, for ``mode="direct"``.
ALL_ORDERS = tuple(range(13))


def default_orders(mode: str, max_order: int = 12) -> tuple[int, ...]:
    """Channels a given mode should emit.

    ``"autocorrelation"`` -> even orders only (odd ones are identically
    zero there); ``"direct"`` -> every order.
    """
    if mode == "autocorrelation":
        return tuple(range(0, max_order + 1, 2))
    return tuple(range(max_order + 1))


def _radial_window(r: np.ndarray, r_max: float, kind: str) -> np.ndarray:
    if kind in (None, "none"):
        return np.ones_like(r)
    if kind == "gauss":
        return np.exp(-0.5 * (r / (r_max / 3.0)) ** 2)
    if kind == "hann":
        return 0.5 * (1.0 + np.cos(np.pi * np.clip(r / r_max, 0.0, 1.0)))
    raise ValueError(f"window must be 'hann', 'gauss' or 'none', got {kind!r}")


@_lru_cache(maxsize=32)
def _polar_grid(n_r: int, r_step: float, n_phi: int):
    """Cached unit polar sampling offsets and the radial bin centres."""
    r = (np.arange(n_r) + 0.5) * r_step
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    return (np.cos(phi)[:, None] * r[None, :],
            np.sin(phi)[:, None] * r[None, :], r)


def _autocorrelation(field, sampling, center, r_max, window, periodic):
    """Circular-windowed, zero-padded autocorrelation about ``center``.

    Zero padding keeps the correlation free of periodic wraparound.  The
    aperture is a disk, so it is rotationally symmetric and cannot inject
    angular structure of its own.
    """
    dx, dy = sampling
    nx, ny = field.shape
    hx, hy = int(np.ceil(r_max / dx)), int(np.ceil(r_max / dy))
    i0, j0 = int(round(center[0] / dx)), int(round(center[1] / dy))
    ii = np.arange(i0 - hx, i0 + hx + 1)
    jj = np.arange(j0 - hy, j0 + hy + 1)
    if periodic:
        ii, jj = np.mod(ii, nx), np.mod(jj, ny)
    else:
        ii, jj = np.clip(ii, 0, nx - 1), np.clip(jj, 0, ny - 1)
    patch = field[np.ix_(ii, jj)]

    ux = (np.arange(patch.shape[0]) - hx) * dx
    uy = (np.arange(patch.shape[1]) - hy) * dy
    rho = np.hypot(ux[:, None], uy[None, :])
    patch = (patch - patch.mean()) * np.where(
        rho < r_max, _radial_window(rho, r_max, window), 0.0)

    n0, n1 = 2 * patch.shape[0], 2 * patch.shape[1]
    spec = np.fft.rfft2(patch, s=(n0, n1))
    acf = np.fft.fftshift(np.fft.irfft2(spec * np.conj(spec), s=(n0, n1)))
    return acf, ((n0 // 2) * dx, (n1 // 2) * dy)


def polar_fft_features(
    field: np.ndarray,
    sampling: tuple[float, float],
    center: tuple[float, float],
    *,
    n_r: int = 100,
    r_step: float = 0.1,
    n_phi: int = 128,
    max_order: int = 12,
    mode: str = "autocorrelation",
    window: str = "hann",
    taper: str = "hann",
    orders: "tuple[int, ...] | None" = None,
    spline_order: int = 3,
    periodic: bool = True,
) -> np.ndarray:
    """Angular-symmetry descriptor about ``center``.

    Parameters
    ----------
    field
        ``(nx, ny)`` image — the full frame, not a pre-windowed crop.
    sampling
        Pixel size ``(dx, dy)`` in Å.
    center
        Window centre ``(cx, cy)`` in Å.
    n_r, r_step
        Radial grid: bin centres ``(arange(n_r) + 0.5) * r_step`` Å.  The
        default spans 0.05 … 9.95 Å, matching the g2 / g3 target grid, so
        the descriptor and the target share one radial axis.
    n_phi
        Angular samples per annulus.  Sampling a *uniform* polar grid is
        what keeps the square pixel lattice from leaking its own 4-fold
        symmetry into orders 4, 8 and 12 at small radius (measured on an
        isotropic field, that leakage drops from 5.5e-2 to 3.7e-8).
    max_order
        Highest angular order kept.
    mode
        ``"autocorrelation"`` (default) or ``"direct"`` — see the module
        docstring.  This also sets the channel count: 7 even channels for
        the autocorrelation, 13 for direct.
    window
        Radial window on the analysis patch: ``"hann"`` (default),
        ``"gauss"`` or ``"none"``.
    taper
        Radial taper applied to the output so every channel falls to zero
        at ``r_max``.  ``"hann"`` by default; ``"none"`` disables it.
    orders
        Explicit channels, overriding the mode default.
    spline_order
        Interpolation order for the polar resampling (3 = bicubic).
    periodic
        Sample the field with periodic wrap (correct for an atomode
        supercell).

    Returns
    -------
    numpy.ndarray
        ``(len(orders), n_r)`` non-negative features.  Use
        :func:`default_orders` to recover which order each row is.
    """
    from scipy.ndimage import map_coordinates

    arr = np.asarray(field, dtype=np.float64)
    dx, dy = sampling
    r_max = float(n_r) * float(r_step)
    if orders is None:
        orders = default_orders(mode, int(max_order))
    orders = tuple(int(o) for o in orders)
    if n_phi <= 2 * max(orders):
        raise ValueError(f"n_phi={n_phi} cannot represent order {max(orders)}")

    if mode == "autocorrelation":
        arr, center = _autocorrelation(arr, sampling, center, r_max, window, periodic)
        wrap = False
    elif mode == "direct":
        wrap = periodic
    else:
        raise ValueError(f"mode must be 'autocorrelation' or 'direct', got {mode!r}")

    ux, uy, r = _polar_grid(int(n_r), float(r_step), int(n_phi))
    vals = map_coordinates(
        arr,
        [((center[0] + ux) / dx).ravel(), ((center[1] + uy) / dy).ravel()],
        order=int(spline_order), mode="grid-wrap" if wrap else "nearest",
    ).reshape(int(n_phi), int(n_r))

    spec = np.abs(np.fft.fft(vals - vals.mean(), axis=0)) / int(n_phi)
    out = spec[list(orders), :]

    scale = float(np.abs(out[0]).mean())
    if scale > 1e-12:
        out = out / scale
    return out * _radial_window(r, r_max, taper)[None, :]


def add_polar_features(
    pairs,
    field_fn,
    sampling: tuple[float, float],
    *,
    key: str = "polar",
    show_progress: bool = True,
    **kwargs,
):
    """Attach :func:`polar_fft_features` to every pair, in place.

    ``field_fn(pair) -> (nx, ny)`` returns the full frame the pair came
    from.  Extra keyword arguments are forwarded to the feature builder.
    """
    total = len(pairs)
    for n, p in enumerate(pairs, 1):
        p[key] = polar_fft_features(field_fn(p), sampling, (p["cx"], p["cy"]), **kwargs)
        if show_progress and (n % 25 == 0 or n == total):
            print(f"\r  angular features: {n}/{total}", end="", flush=True)
    if show_progress and total:
        print()
    return pairs
