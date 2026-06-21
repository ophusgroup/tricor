"""Window / envelope weighting for local ptychographic g2 / g3 targets.

A *local* correlation function is measured over a soft analysis window: a
**circular** (radial) Hann taper in the imaging plane ``(x, y)`` and a
Gaussian along the beam direction ``z``, centred on a chosen slice ``z0``.
Every atom *i* carries a scalar weight

    w_i = H(r_i) · G(z_i - z0),   r_i = hypot(x_i - cx, y_i - cy)

so atoms near the window edge contribute little and the recovered g2 / g3
describes the *same* soft region that the blurred potential slice shows.
The window is **isotropic** (a disk of radius ``side / 2``, zero in the
four corners), so rotating the input image about the window centre leaves
g2 / g3 unchanged — rotation is a free augmentation of the *input* only.
The matching asymptote-to-1 normalisation lives in
:mod:`tricor.ptycho.correlations`.

Windows are assumed to lie fully inside the (orthorhombic, periodic)
cell — the draggable widget and the sliding-window sampler both keep the
support away from the box faces, so plain displacements (no minimum
image) are exact and fast.  This also bounds the usable ``sigma_z`` by
the z extent of the cell.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "WindowSpec",
    "hann_1d",
    "hann_radial",
    "hann_window_xy",
    "gaussian_z",
    "window_weights",
    "windowed_image",
]


@dataclasses.dataclass(frozen=True)
class WindowSpec:
    """Geometry of the soft analysis window.

    Parameters
    ----------
    center_xy
        In-plane centre ``(cx, cy)`` of the Hann window, in Å.
    side
        Hann window side length ``L`` (Å).  The window is a *disk* of
        radius ``L / 2`` (zero in the corners of the ``L × L`` box) and the
        useful correlation range is ``r <= L / 2``.
    z0
        Centre of the Gaussian depth weight (Å).
    sigma_z
        Standard deviation of the Gaussian depth weight (Å) — the
        pseudo-ptychographic depth resolution.
    z_support_sigmas
        Half-extent of the z support, in units of ``sigma_z``.  Atoms
        beyond ``z0 ± z_support_sigmas · sigma_z`` are dropped.
    """

    center_xy: tuple[float, float]
    side: float
    z0: float
    sigma_z: float
    z_support_sigmas: float = 3.0

    @property
    def z_support(self) -> float:
        """Half-height of the z support window (Å)."""
        return self.z_support_sigmas * self.sigma_z

    @property
    def r_window(self) -> float:
        """Largest correlation radius the window can describe (Å)."""
        return 0.5 * self.side


def hann_1d(d: np.ndarray, side: float) -> np.ndarray:
    """Hann taper, 1 at ``d = 0`` and 0 at ``|d| >= side / 2`` (compact)."""
    d = np.asarray(d, dtype=np.float64)
    half = 0.5 * side
    w = np.zeros_like(d)
    inside = np.abs(d) < half
    w[inside] = 0.5 * (1.0 + np.cos(2.0 * np.pi * d[inside] / side))
    return w


def hann_radial(dr: np.ndarray, side: float) -> np.ndarray:
    """Circular Hann window: 1 at ``dr = 0``, 0 at ``dr >= side / 2``.

    ``dr`` is the in-plane distance from the window centre.  The support is
    the disk of radius ``side / 2`` — the four corners of the enclosing
    ``side × side`` box are exactly zero, making the window isotropic.
    """
    dr = np.asarray(dr, dtype=np.float64)
    half = 0.5 * side
    w = np.zeros_like(dr)
    inside = dr < half
    w[inside] = 0.5 * (1.0 + np.cos(np.pi * dr[inside] / half))
    return w


def hann_window_xy(x, y, center_xy: tuple[float, float], side: float) -> np.ndarray:
    """Circular Hann window evaluated at in-plane positions ``(x, y)``."""
    cx, cy = center_xy
    dx = np.asarray(x, dtype=np.float64) - cx
    dy = np.asarray(y, dtype=np.float64) - cy
    return hann_radial(np.hypot(dx, dy), side)


def gaussian_z(z, z0: float, sigma_z: float) -> np.ndarray:
    """Gaussian depth weight centred on ``z0`` with width ``sigma_z``."""
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * ((z - z0) / sigma_z) ** 2)


def window_weights(
    positions: np.ndarray,
    spec: WindowSpec,
    box: np.ndarray | None = None,
    atom_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Per-atom window weight ``w_i`` (0 outside the support).

    Parameters
    ----------
    positions
        ``(N, 3)`` Cartesian atom positions (Å).
    spec
        The :class:`WindowSpec` describing the window.
    box
        Orthorhombic cell lengths ``(Lx, Ly, Lz)``.  When given, the
        displacement from the window centre uses the minimum-image
        convention, so a window placed near (or across) a periodic face
        wraps correctly.
    atom_scale
        Optional ``(N,)`` per-atom multiplier — e.g. the per-species
        scattering power from
        :func:`tricor.ptycho.potential.scattering_power`, so the weighted
        g2 / g3 matches the scattering-weighted potential.  ``None``
        gives an unweighted (count) window.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` non-negative weights.  Atoms outside the ``z`` support
        are forced to exactly zero so they can be masked away cheaply.
    """
    positions = np.asarray(positions, dtype=np.float64)
    cx, cy = spec.center_xy
    dx = positions[:, 0] - cx
    dy = positions[:, 1] - cy
    dz = positions[:, 2] - spec.z0
    if box is not None:
        box = np.asarray(box, dtype=np.float64)
        dx -= np.round(dx / box[0]) * box[0]
        dy -= np.round(dy / box[1]) * box[1]
        dz -= np.round(dz / box[2]) * box[2]
    w = hann_radial(np.hypot(dx, dy), spec.side)
    w = w * np.exp(-0.5 * (dz / spec.sigma_z) ** 2)
    if atom_scale is not None:
        w = w * np.asarray(atom_scale, dtype=np.float64)
    # Hard-clip the long Gaussian tail to the declared support so the
    # window has finite extent (keeps the random-catalogue support and
    # the data support identical).
    w[np.abs(dz) > spec.z_support] = 0.0
    return w


def windowed_image(
    array: np.ndarray,
    sampling: tuple[float, float],
    center: tuple[float, float],
    side: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Rotated, circular-Hann-windowed crop of a periodic 2D field.

    Sample ``array`` (a periodic field with pixel size ``sampling = (dx,
    dy)`` Å) on a ``side × side`` grid centred at ``center = (cx, cy)`` Å
    and rotated ``angle_deg`` (CCW) about that centre, using a smooth,
    periodic, bicubic (cubic-spline) interpolator.  The crop is
    mean-normalised, multiplied by the circular Hann window, and the
    background outside the disk is set to 1::

        im = crop / mean(crop) * window + (1 - window)

    Because the window is a disk, rotating the crop only re-samples the
    *input*; the matching g2 / g3 target is unchanged — this is the
    rotation augmentation the dense network is trained against.
    """
    from scipy.ndimage import map_coordinates

    arr = np.asarray(array, dtype=np.float64)
    dx, dy = sampling
    cx, cy = center
    nx_half = int(round((0.5 * side) / dx))
    ny_half = int(round((0.5 * side) / dy))
    u = (np.arange(-nx_half, nx_half) + 0.5) * dx
    v = (np.arange(-ny_half, ny_half) + 0.5) * dy
    uu, vv = np.meshgrid(u, v, indexing="ij")
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Source positions (Å) of the rotated local grid, then -> pixel index.
    su = cx + cos_t * uu - sin_t * vv
    sv = cy + sin_t * uu + cos_t * vv
    crop = map_coordinates(arr, [su / dx, sv / dy], order=3, mode="grid-wrap")
    window = hann_radial(np.hypot(uu, vv), side)
    mean = float(np.mean(crop))
    norm = crop / mean if mean > 1e-12 else crop
    return norm * window + (1.0 - window)
