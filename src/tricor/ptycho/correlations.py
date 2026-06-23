"""Local, window-weighted g2 / g3 correlation functions.

These are the *targets* for the ptychography → local-structure ML task.
Given an atomic structure and a soft :class:`~tricor.ptycho.weighting.WindowSpec`,
they return correlation functions that

* weight every atom by the window envelope ``w_i`` (edge atoms barely
  contribute; a triplet contributes ``w_i · w_j · w_k``), and
* **asymptote to 1** at large radius for *any* window / blur, by
  dividing out the window+geometry envelope with a uniform random
  catalogue and then scaling the far field to 1.

Normalisation strategy
----------------------
For a Poisson (structure-free) set of points carrying the same window
weight, the weighted pair / triplet histogram is exactly the window's
auto-correlation envelope.  Dividing the data histogram ``DD`` by the
random-catalogue histogram ``RR`` removes that envelope (and all edge
effects), leaving a flat function for random data; a final far-field
rescale pins it to 1.  This mirrors tricor's own far-field amplitude
normalisation in :mod:`tricor.g3`, generalised to a spatially varying
weight.

The ideal-gas g3 denominator factorises exactly into
``RR(r01)·RR(r02)·sin φ`` (independent random neighbours), so only the
cheap pair envelope ``RR`` is built by Monte-Carlo — analytic in angle,
matching tricor's reduced-g3 convention.  ``RR`` depends only on the
window *shape* (``side``, ``sigma_z``) and the binning — not on its
position or the structure — so it is cached and reused across every
window and every cell (the key to cheap sliding-window training-pair
generation).

v1 scope: a single (species-agnostic) channel — every atom counts
equally inside the window.  Per-species channels and scattering-power
weighting are planned follow-ups.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .weighting import WindowSpec, window_weights

__all__ = ["LocalCorrelations", "local_correlations", "clear_random_cache"]

_EPS = 1e-12
# Cache of random-catalogue histograms, keyed by window shape + binning.
_RANDOM_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


@dataclasses.dataclass
class LocalCorrelations:
    """Result of :func:`local_correlations`.

    Attributes
    ----------
    r
        ``(num_r,)`` radial bin centres (Å).
    phi_deg
        ``(num_phi,)`` angular bin centres (degrees, 0–180).
    g2
        ``(num_r,)`` window-weighted pair correlation, → 1 at large r.
    g3
        ``(num_r, num_r, num_phi)`` weighted rooted three-body
        correlation ``g3(r01, r02, phi)``, → 1 at large radius.
    g3_slice
        ``(num_phi, num_r)`` integrated-g3 target: ``r01`` pinned to the
        nearest-neighbour band, integrated → a (angle, distance) map,
        → 1 at large r.  Oriented like tricor's plot slice (phi rows).
    nn_band
        ``(r_lo, r_hi)`` of the r01 integration band (Å).
    n_window
        Number of atoms with non-negligible weight inside the window.
    weight_sum
        Sum of per-atom weights (effective atom count).
    """

    r: np.ndarray
    phi_deg: np.ndarray
    g2: np.ndarray
    g3: np.ndarray
    g3_slice: np.ndarray
    nn_band: tuple[float, float]
    n_window: int
    weight_sum: float


def clear_random_cache() -> None:
    """Drop all cached random-catalogue histograms."""
    _RANDOM_CACHE.clear()


def _orthorhombic_box(atoms) -> np.ndarray:
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    off = cell - np.diag(np.diag(cell))
    if np.abs(off).max() > 1e-6 * max(1.0, np.abs(cell).max()):
        raise ValueError("ptycho.correlations requires an orthorhombic cell.")
    return np.diag(cell)


def _pair_hist(
    pos: np.ndarray, w: np.ndarray, num_r: int, r_step: float, box: np.ndarray | None = None
) -> np.ndarray:
    """Weighted pair histogram DD(r) = sum_{i!=j} w_i w_j 1[bin(r_ij)=r].

    ``box`` (orthorhombic lengths) enables the minimum-image convention so
    windows that wrap across the periodic boundary are handled correctly.
    """
    out = np.zeros(num_r, dtype=np.float64)
    r_max = num_r * r_step
    n = len(pos)
    for i in range(n):
        d = pos - pos[i]
        if box is not None:
            d -= np.round(d / box) * box
        rij = np.sqrt(np.einsum("ij,ij->i", d, d))
        m = (rij > 1e-9) & (rij < r_max)
        if not m.any():
            continue
        rb = (rij[m] / r_step).astype(np.intp)
        out += np.bincount(rb, weights=w[i] * w[m], minlength=num_r)
    return out


def _triplet_hist(
    pos: np.ndarray,
    w: np.ndarray,
    num_r: int,
    r_step: float,
    num_phi: int,
    box: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted rooted-triplet histogram DDD(r01, r02, phi).

    Rooted on each atom i: neighbours j, k within ``r_max`` contribute
    ``w_i · w_j · w_k`` to bin ``(r_ij, r_ik, angle(j,i,k))``.  ``box``
    enables the minimum-image convention (wraparound windows).
    """
    flat = np.zeros(num_r * num_r * num_phi, dtype=np.float64)
    r_max = num_r * r_step
    phi_step = np.pi / num_phi
    n = len(pos)
    for i in range(n):
        d = pos - pos[i]
        if box is not None:
            d -= np.round(d / box) * box
        r2 = np.einsum("ij,ij->i", d, d)
        m = (r2 > 1e-12) & (r2 < r_max * r_max)
        if m.sum() < 2:
            continue
        v = d[m]
        rr = np.sqrt(r2[m])
        wn = w[m]
        rb = (rr / r_step).astype(np.intp)
        cos_phi = np.clip((v @ v.T) / np.outer(rr, rr), -1.0, 1.0)
        pb = (np.arccos(cos_phi) / phi_step).astype(np.intp)
        np.clip(pb, 0, num_phi - 1, out=pb)
        wt = w[i] * np.outer(wn, wn)
        np.fill_diagonal(wt, 0.0)  # drop j == k
        lin = (rb[:, None] * num_r + rb[None, :]) * num_phi + pb
        flat += np.bincount(lin.ravel(), weights=wt.ravel(), minlength=flat.size)
    return flat.reshape(num_r, num_r, num_phi)


def _random_histograms(
    spec: WindowSpec, num_r: int, r_step: float, num_phi: int, n_random: int, rng_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Window-matched ideal-gas envelopes ``RR(r)`` and ``RRR(r01,r02,phi)``.

    A uniform (structure-free) catalogue carrying the *same* window
    weights gives the exact normalisation: dividing the data histograms
    by these makes g2 / g3 → 1 for an uncorrelated structure.  ``RRR``
    captures the window's angular anisotropy (a Gaussian-in-z, Hann-in-xy
    window is not isotropic), which an analytic ``sin φ`` cannot.

    Translation-invariant in the window position, so cached by window
    *shape* + binning and reused across windows / structures (the
    triplet build is the one-time cost).
    """
    key = (
        round(spec.side, 4),
        spec.z_mode,
        round(spec.z_support, 4),
        round(spec.sigma_z, 4),
        round(spec.z_support_sigmas, 4),
        num_r,
        round(r_step, 6),
        num_phi,
        n_random,
        rng_seed,
    )
    cached = _RANDOM_CACHE.get(key)
    if cached is not None:
        return cached

    rng = np.random.default_rng(rng_seed)
    half = 0.5 * spec.side
    zr = spec.z_support
    rp = np.empty((n_random, 3), dtype=np.float64)
    rp[:, 0] = rng.uniform(-half, half, n_random)
    rp[:, 1] = rng.uniform(-half, half, n_random)
    rp[:, 2] = rng.uniform(-zr, zr, n_random)
    rw = window_weights(rp, spec.centered())
    keep = rw > rw.max() * 1e-4
    rp, rw = rp[keep], rw[keep]

    rr = _pair_hist(rp, rw, num_r, r_step)
    rrr = _triplet_hist(rp, rw, num_r, r_step, num_phi)
    _RANDOM_CACHE[key] = (rr, rrr)
    return rr, rrr


def _kde(values: np.ndarray, sigma) -> np.ndarray:
    """Gaussian KDE smoothing of a histogram (``mode='constant'``).

    Applied to **both** the numerator and the denominator of g2 / g3
    before dividing, so the kernel fills empty bins (no 0-count bins),
    the ratio is finite everywhere, and the boundary attenuation cancels
    between numerator and denominator — keeping r = 0, r = r_max and
    φ = 0 / 180° well defined.
    """
    sig = sigma if isinstance(sigma, tuple) else (sigma,)
    arr = np.asarray(values, dtype=np.float64)
    if not any(s > 0 for s in sig):
        return arr
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(arr, sig, mode="constant", cval=0.0)


def local_correlations(
    atoms,
    window: WindowSpec,
    *,
    r_max: float = 10.0,
    r_step: float = 0.1,
    phi_num_bins: int = 36,
    pair_peak: float | None = None,
    nn_band: tuple[float, float] = (0.75, 1.25),
    atom_scale: np.ndarray | None = None,
    n_random: int | None = None,
    rng_seed: int = 0,
    smooth_random: bool = True,
    smooth_sigma_r: float = 0.1,
    smooth_sigma_phi_deg: float = 5.0,
) -> LocalCorrelations:
    """Window-weighted g2, g3 and the integrated-g3 slice for one window.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` in an orthorhombic, periodic cell.
    window
        The :class:`~tricor.ptycho.weighting.WindowSpec`.
    r_max, r_step, phi_num_bins
        Binning, matching tricor's g3 conventions.  ``r_max`` should not
        exceed ``window.side / 2``.  For a *local* window the angular
        statistics are sparse, so ``phi_num_bins`` defaults to 36 (5°);
        18 (10°) and 90 (2°) are also reasonable.  ``r_step`` of 0.1 Å is
        kept and the sparse counts are handled by the smoothing below.
    smooth_sigma_r, smooth_sigma_phi_deg
        Kernel-density Gaussian smoothing widths applied to the final
        g2 / g3 (in Å and degrees).  Edge-aware (see
        :func:`_normalized_smooth`), so the r = 0 / r_max and φ = 0 / 180°
        bins stay unbiased.  Set to 0 to disable.
    pair_peak
        Nearest-neighbour bond distance (Å) used to place the r01 band
        for the integrated-g3 slice.  If ``None``, taken as the location
        of the first g2 peak.
    nn_band
        Multipliers ``(lo, hi)`` on ``pair_peak`` for the r01 band.
    atom_scale
        Optional ``(N,)`` per-atom weight folded into the data histograms
        — typically the per-species scattering power from
        :func:`tricor.ptycho.potential.scattering_power`, so the target
        is the scattering-weighted g2 / g3 the potential encodes.  The
        random envelope stays geometric; the far-field rescale absorbs the
        overall scattering constant, so the result still → 1.
    n_random
        Uniform-catalogue size for the normalisation (cached per window
        shape).  ``None`` auto-sizes it to a modest density inside the
        window support, which keeps the one-time ``RRR`` cost bounded;
        larger values give a cleaner (less noisy) cached denominator.
    smooth_random
        Lightly smooth the random catalogue histograms (their true value
        is smooth) to reduce Monte-Carlo noise in the denominator.

    Returns
    -------
    LocalCorrelations
    """
    box = _orthorhombic_box(atoms)
    num_r = int(round(r_max / r_step))
    r_centers = (np.arange(num_r) + 0.5) * r_step
    phi_edges = np.linspace(0.0, np.pi, phi_num_bins + 1)
    phi_deg = np.rad2deg(0.5 * (phi_edges[:-1] + phi_edges[1:]))

    pos = np.asarray(atoms.positions, dtype=np.float64)
    w = window_weights(pos, window, box=box, atom_scale=atom_scale)
    keep = w > w.max() * 1e-4 if w.max() > 0 else np.zeros(len(w), bool)
    pos_k, w_k = pos[keep], w[keep]

    if n_random is None:
        # Size the catalogue to a modest density inside the window support;
        # the triplet RRR build is the (cached, one-time) cost.
        v_box = window.side * window.side * 2.0 * window.z_support
        n_random = int(np.clip(round(0.10 * v_box), 1500, 5000))

    dd = _pair_hist(pos_k, w_k, num_r, r_step, box=box)
    ddd = _triplet_hist(pos_k, w_k, num_r, r_step, phi_num_bins, box=box)
    rr, rrr = _random_histograms(window, num_r, r_step, phi_num_bins, n_random, rng_seed)

    # Denoise the Monte-Carlo ideal envelopes (their true value is smooth)
    # before they enter the denominator.
    if smooth_random:
        from scipy.ndimage import gaussian_filter, gaussian_filter1d

        rr = gaussian_filter1d(rr, 1.0, mode="nearest")
        rrr = gaussian_filter(rrr, (1.0, 1.0, 1.0), mode="nearest")

    # KDE smoothing widths, in bins.
    sig_r = smooth_sigma_r / r_step
    sig_phi = smooth_sigma_phi_deg / (180.0 / phi_num_bins)

    # KDE is applied to BOTH the numerator and the (window-matched ideal)
    # denominator, then divided — the kernel fills empty bins so the ratio
    # is finite everywhere (r = 0, r = r_max, φ = 0/180°).  Dividing by the
    # Monte-Carlo RRR (not an analytic sin φ) removes the window's angular
    # anisotropy.  The overall level is set by the total pair / triplet
    # counts so the result → 1 for a *random* structure while an ordered
    # one keeps its tall sharp peaks (a far-field rescale would squash a
    # crystal, whose far field is not 1).
    # --- g2 ---
    g2 = _kde(dd, (sig_r,)) / (_kde(rr, (sig_r,)) + _EPS)
    g2 = g2 * (float(rr.sum()) / max(float(dd.sum()), _EPS))

    # --- g3 ---
    g3 = _kde(ddd, (sig_r, sig_r, sig_phi)) / (_kde(rrr, (sig_r, sig_r, sig_phi)) + _EPS)
    g3 = g3 * (float(rrr.sum()) / max(float(ddd.sum()), _EPS))

    # --- integrated slice: pin r01 to the NN band, integrate counts and
    # the matched envelope over the band, *then* divide. ---
    if pair_peak is None:
        lo_i = int(0.3 / r_step)  # ignore the r → 0 region
        pair_peak = float(r_centers[lo_i + int(np.argmax(g2[lo_i:]))])
    band = (nn_band[0] * pair_peak, nn_band[1] * pair_peak)
    band_mask = (r_centers >= band[0]) & (r_centers < band[1])
    if not band_mask.any():
        band_mask[int(np.argmin(np.abs(r_centers - pair_peak)))] = True
    ddd_band = ddd[band_mask].sum(axis=0)  # (r02, phi)
    rrr_band = rrr[band_mask].sum(axis=0)
    g3_slice = _kde(ddd_band, (sig_r, sig_phi)) / (_kde(rrr_band, (sig_r, sig_phi)) + _EPS)
    g3_slice = g3_slice * (float(rrr_band.sum()) / max(float(ddd_band.sum()), _EPS))
    g3_slice = g3_slice.T  # -> (phi, r02), like tricor's plot slice

    return LocalCorrelations(
        r=r_centers,
        phi_deg=phi_deg,
        g2=g2,
        g3=g3,
        g3_slice=g3_slice,
        nn_band=band,
        n_window=int(keep.sum()),
        weight_sum=float(w_k.sum()),
    )
