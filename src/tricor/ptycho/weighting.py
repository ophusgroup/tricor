"""Window / envelope weighting for local ptychographic g2 / g3 targets.

A *local* correlation function is measured over a soft analysis window: a
2D Hann taper in the imaging plane ``(x, y)`` and a Gaussian along the
beam direction ``z``, centred on a chosen slice ``z0``.  Every atom *i*
carries a scalar weight

    w_i = H(x_i - cx) · H(y_i - cy) · G(z_i - z0)

so atoms near the window edge contribute little and the recovered g2 / g3
describes the *same* soft region that the blurred potential slice shows.
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
    "hann_window_xy",
    "gaussian_z",
    "window_weights",
]


@dataclasses.dataclass(frozen=True)
class WindowSpec:
    """Geometry of the soft analysis window.

    Parameters
    ----------
    center_xy
        In-plane centre ``(cx, cy)`` of the Hann window, in Å.
    side
        Hann window side length ``L`` (Å).  The window is zero outside the
        ``L × L`` square and the useful correlation range is ``r <= L / 2``.
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


def hann_window_xy(x, y, center_xy: tuple[float, float], side: float) -> np.ndarray:
    """Separable 2D Hann window evaluated at atom positions ``(x, y)``."""
    cx, cy = center_xy
    return hann_1d(np.asarray(x, dtype=np.float64) - cx, side) * hann_1d(
        np.asarray(y, dtype=np.float64) - cy, side
    )


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
    w = hann_1d(dx, spec.side) * hann_1d(dy, spec.side)
    w = w * np.exp(-0.5 * (dz / spec.sigma_z) ** 2)
    if atom_scale is not None:
        w = w * np.asarray(atom_scale, dtype=np.float64)
    # Hard-clip the long Gaussian tail to the declared support so the
    # window has finite extent (keeps the random-catalogue support and
    # the data support identical).
    w[np.abs(dz) > spec.z_support] = 0.0
    return w
