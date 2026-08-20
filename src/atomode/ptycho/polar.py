"""Rotation-invariant angular-symmetry features of a local image patch.

Reduce the neighbourhood around a window centre to a compact
``(max_order + 1, n_r)`` descriptor of its local angular symmetry.  For
each order ``m`` and each annulus ``r`` about the centre::

    F_m(r) = sum_pixels  I(x, y) . W(x, y) . exp(i m atan2(y, x))

and the descriptor keeps ``|F_m(r)|``.  ``W`` is a radial window, 1 at
the centre and falling to 0 at ``r_max``, applied to **every** order
``m = 0 … max_order``, so the descriptor is weighted toward the centre of
the window and goes smoothly to zero at its edge.

Row ``m`` is the amplitude of ``m``-fold angular modulation at each
radius: row 3 responds to an sp2 (graphene-like) motif, row 4 to a square
lattice, row 6 to a close-packed one, each with its harmonics.

Rotation invariance
-------------------
Dropping the phase of ``F_m`` makes the descriptor rotation invariant.
Rotating the structure by ``theta`` multiplies ``F_m`` by
``exp(i m theta)``, which ``|.|`` removes, so the complex annular
integral picks up ``m``-fold content at the same strength whatever the
local orientation of the motif.  Nothing needs to be aligned or
detected first, and the descriptor pairs directly with the
rotation-invariant g2 / g3 targets, so no rotation augmentation is
needed.

Real space, not diffraction space
---------------------------------
The transform is applied to the image rather than to its diffractogram
deliberately.  ``|FFT|`` of a real image is centrosymmetric (Friedel), so
its angular profile has period 180 deg and every odd order vanishes
identically (verified: odd/even ~ 5e-16).  A 3-fold sp2 motif then shows
up only as a 6-fold diffractogram, indistinguishable from a genuinely
6-fold one.  Real space keeps the odd orders.

The transform is evaluated about the **window centre only** — the one
point that is always known, in simulation and in experiment alike.
"""

from __future__ import annotations

from functools import lru_cache as _lru_cache

import numpy as np

__all__ = [
    "angular_symmetry_kernel",
    "polar_fft_features",
    "add_polar_features",
]


@_lru_cache(maxsize=32)
def angular_symmetry_kernel(
    sampling: tuple[float, float],
    n_r: int,
    r_step: float,
    max_order: int,
    window: str = "hann",
    r_smooth: float = 0.0,
) -> dict:
    """Cached patch geometry for :func:`polar_fft_features`.

    Builds, once per (pixel size, radial grid, order, window) combination:
    the pixel offsets covering the disk of radius ``n_r * r_step``, their
    radial bin assignment, and the complex factors
    ``W(rho) * exp(-i m phi)`` for every order.

    Radial binning is linear (cloud-in-cell): each pixel splits between
    the two nearest radial bins, so the profile varies smoothly rather
    than stepping as pixels cross a bin edge.

    ``window`` is ``"hann"`` (1 at the centre, 0 at ``r_max``), ``"none"``
    for a flat top-hat, or ``"gauss"`` for a Gaussian of sigma
    ``r_max / 3``.
    """
    dx, dy = sampling
    r_max = n_r * r_step
    nhx, nhy = int(np.ceil(r_max / dx)), int(np.ceil(r_max / dy))
    di = np.arange(-nhx, nhx + 1)
    dj = np.arange(-nhy, nhy + 1)
    ddx = di[:, None] * dx
    ddy = dj[None, :] * dy
    rho = np.hypot(ddx, ddy)
    phi = np.arctan2(ddy, ddx)
    keep = rho < r_max

    di_k = np.ascontiguousarray(np.broadcast_to(di[:, None], rho.shape)[keep])
    dj_k = np.ascontiguousarray(np.broadcast_to(dj[None, :], rho.shape)[keep])
    rho_k, phi_k = rho[keep], phi[keep]

    if window == "hann":
        w_rad = 0.5 * (1.0 + np.cos(np.pi * rho_k / r_max))
    elif window == "gauss":
        w_rad = np.exp(-0.5 * (rho_k / (r_max / 3.0)) ** 2)
    elif window == "none":
        w_rad = np.ones_like(rho_k)
    else:
        raise ValueError(f"window must be 'hann', 'gauss' or 'none', got {window!r}")

    n_orders = int(max_order) + 1
    orders = np.arange(n_orders)[:, None]
    # Window folded into the complex factor, so it multiplies every order.
    e_mw = np.exp(-1j * orders * phi_k[None, :]) * w_rad[None, :]

    # Radial kernel: triangular of half-width `h`.  h == r_step reduces to
    # plain linear (cloud-in-cell) binning between the two nearest bins.
    # A wider kernel puts more pixels in each annulus, which samples the
    # angle more uniformly and suppresses the square-pixel-lattice
    # anisotropy; it costs radial resolution the pixel size cannot support
    # anyway.
    h = float(r_smooth) if r_smooth and r_smooth > r_step else float(r_step)
    span = int(np.ceil(h / r_step))
    t = rho_k / r_step - 0.5
    kc = np.round(t).astype(np.intp)

    bins = []
    for off in range(-span, span + 1):
        k = kc + off
        w = np.maximum(0.0, 1.0 - np.abs(t - k) * r_step / h)
        bins.append((k, w))
    tot = np.sum([w for _, w in bins], axis=0)
    tot = np.maximum(tot, 1e-12)
    bins = [(k, w / tot) for k, w in bins]

    parts = []
    wsum = np.zeros(n_r, dtype=np.float64)
    for k, cic in bins:
        ok = (k >= 0) & (k < n_r)
        idx = (np.arange(n_orders)[:, None] * n_r + k[None, :])[:, ok].ravel()
        parts.append((np.ascontiguousarray(idx),
                      np.ascontiguousarray(e_mw[:, ok] * cic[ok][None, :]),
                      np.ascontiguousarray(ok)))
        wsum += np.bincount(k[ok], (cic * w_rad)[ok], minlength=n_r)[:n_r]

    return dict(di=di_k, dj=dj_k, parts=parts, n_r=int(n_r),
                n_orders=n_orders, wsum=np.maximum(wsum, 1e-12),
                r=(np.arange(n_r) + 0.5) * r_step)


def polar_fft_features(
    field: np.ndarray,
    sampling: tuple[float, float],
    center: tuple[float, float],
    *,
    n_r: int = 100,
    r_step: float = 0.1,
    max_order: int = 12,
    window: str = "hann",
    r_smooth: float | None = None,
    normalize: bool = False,
    scale: str = "mean",
    periodic: bool = True,
) -> np.ndarray:
    """Angular-symmetry descriptor about ``center``.

    ``|sum I . W . exp(i m phi)|`` per annulus, for orders
    ``0 … max_order``, evaluated directly on the native pixel grid (no
    interpolation; every pixel contributes at its exact angle).

    Parameters
    ----------
    field
        ``(nx, ny)`` image — the full frame, not a pre-windowed crop.
    sampling
        Pixel size ``(dx, dy)`` in Å.
    center
        Window centre ``(cx, cy)`` in Å.  Snapped to the nearest pixel.
    n_r, r_step
        Radial grid: bin centres ``(arange(n_r) + 0.5) * r_step`` Å.  The
        default matches the g2 / g3 target grid (0.05 … 9.95 Å), so the
        feature rows and the target share one radial axis.
    max_order
        Highest angular order kept; the output has ``max_order + 1`` rows
        (12 → 13 rows, orders 0 … 12).
    window
        Radial weight applied to every order: ``"hann"`` (default, 1 at
        the centre falling to 0 at ``r_max``), ``"gauss"``, or ``"none"``.
    r_smooth
        Half-width (Å) of the triangular radial kernel.  ``None`` (default)
        uses two pixels.  A thin annulus holds few pixels and samples the
        angle unevenly on a square lattice, which shows up as rotation
        anisotropy (~20% at one bin, ~0.5% at two pixels).  Widening it
        costs radial resolution the pixel size cannot support anyway.
        ``0`` gives plain linear binning between the two nearest bins.
    normalize
        Divide each annulus by its summed window weight.  This *removes*
        the window taper and returns a per-annulus mean, so it is off by
        default: the point of the window is that the descriptor is
        weighted toward the centre and vanishes at the edge.
    scale
        Overall scaling, applied after the annular sums.  ``"mean"``
        (default) divides by the mean field value in the patch, making the
        descriptor independent of overall image contrast; ``"none"``
        leaves the raw sums.
    periodic
        Sample the field with periodic wrap (correct for an atomode
        supercell).  ``False`` clamps at the edges.

    Returns
    -------
    numpy.ndarray
        ``(max_order + 1, n_r)`` non-negative features.
    """
    arr = np.asarray(field, dtype=np.float64)
    dx, dy = sampling
    # Default the radial kernel to two pixels: enough to sample the angle
    # uniformly (rotation error ~0.5% instead of ~20%) and finer than the
    # in-plane resolution the field carries anyway.
    rs = 2.0 * max(float(dx), float(dy)) if r_smooth is None else float(r_smooth)
    kernel = angular_symmetry_kernel((float(dx), float(dy)), int(n_r),
                                     float(r_step), int(max_order), window, rs)

    nx, ny = arr.shape
    i0 = int(round(center[0] / dx))
    j0 = int(round(center[1] / dy))
    ii = i0 + kernel["di"]
    jj = j0 + kernel["dj"]
    if periodic:
        ii = np.mod(ii, nx)
        jj = np.mod(jj, ny)
    else:
        ii = np.clip(ii, 0, nx - 1)
        jj = np.clip(jj, 0, ny - 1)
    vals = arr[ii, jj]

    n_orders, n_r_ = kernel["n_orders"], kernel["n_r"]
    acc = np.zeros(n_orders * n_r_, dtype=np.complex128)
    for idx, e_mw, ok in kernel["parts"]:
        a = e_mw * vals[ok][None, :]
        acc += (np.bincount(idx, a.real.ravel(), minlength=n_orders * n_r_)
                + 1j * np.bincount(idx, a.imag.ravel(), minlength=n_orders * n_r_))

    out = np.abs(acc.reshape(n_orders, n_r_))
    if normalize:
        out = out / kernel["wsum"][None, :]
    if scale == "mean":
        m = float(vals.mean())
        if abs(m) > 1e-12:
            out = out / m
    elif scale != "none":
        raise ValueError(f"scale must be 'mean' or 'none', got {scale!r}")
    return out


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

    Parameters
    ----------
    pairs
        List from :func:`~atomode.ptycho.sliding_window_pairs` or
        :func:`~atomode.ptycho.sliding_window_pairs_hrtem`.
    field_fn
        ``field_fn(pair) -> (nx, ny)`` full frame the pair came from.
    sampling
        Pixel size ``(dx, dy)`` in Å of the field.
    key
        Pair key to write the ``(max_order + 1, n_r)`` array to.
    **kwargs
        Forwarded to :func:`polar_fft_features`.
    """
    total = len(pairs)
    for n, p in enumerate(pairs, 1):
        p[key] = polar_fft_features(field_fn(p), sampling, (p["cx"], p["cy"]), **kwargs)
        if show_progress and (n % 25 == 0 or n == total):
            print(f"\r  angular features: {n}/{total}", end="", flush=True)
    if show_progress and total:
        print()
    return pairs
