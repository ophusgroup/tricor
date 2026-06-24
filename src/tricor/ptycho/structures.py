"""Spatially graded supercells (order varying along one axis).

Build a cell whose disorder varies along an axis by giving the Voronoi
seed density a linear profile: high seed density (small grains) at one
end, low seed density (large grains) at the other.  The whole cell is a
single Voronoi tessellation followed by a single FIRE relaxation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["graded_supercell"]


def _seed_density(grain_size: float) -> float:
    """Voronoi seed density (seeds / Å³) for a given grain size."""
    radius = max(0.5 * grain_size, 2.0)
    return 1.0 / ((4.0 / 3.0) * np.pi * radius ** 3)


def graded_supercell(
    reference,
    shell,
    *,
    cell_dim=(50.0, 250.0, 50.0),
    axis: int = 1,
    grain_min: float = 5.0,
    grain_max: float = 60.0,
    profile: str = "grain_size",
    num_steps: int = 150,
    displacement_sigma: float = 0.04,
    weights: dict | None = None,
    rng_seed: int = 0,
    r_max: float = 10.0,
    r_step: float = 0.1,
    phi_num_bins: int = 36,
    max_seeds: int = 8000,
    show_progress: bool = True,
):
    """Generate a supercell with the Voronoi seed density graded along ``axis``.

    The seed density varies linearly from the density implied by
    ``grain_min`` (small grains → disordered) at the ``axis = 0`` face to
    that of ``grain_max`` (large grains → ordered) at the ``axis = L``
    face.  One Voronoi tessellation, one FIRE relaxation.

    Parameters
    ----------
    reference
        Reference crystal (ASE ``Atoms``).
    shell
        :class:`~tricor.CoordinationShellTarget` for ``reference``.
    cell_dim
        Box ``(Lx, Ly, Lz)`` in Å.
    axis
        Axis along which order increases (0=x, 1=y, 2=z).
    grain_min, grain_max
        Grain size (Å) at the disordered and ordered ends.  For a sharp
        ordered end, ``grain_max`` should be at least the cell thickness
        along the beam so a window sees a single crystal.
    profile
        ``"grain_size"`` ramps the grain size linearly (even visual
        gradient, strong end-to-end contrast); ``"density"`` ramps the
        seed density linearly (size change concentrated at the sparse
        end).
    num_steps, displacement_sigma, weights, r_max, r_step, phi_num_bins
        :meth:`~tricor.Supercell.generate` settings for the single
        relaxation.
    max_seeds
        Upper bound on the number of seeds (caps the dense end's cost
        while preserving the linear density profile).

    Returns
    -------
    ase.Atoms
        The graded cell (orthorhombic, periodic).
    """
    import tricor as tc

    box = np.array([float(v) for v in cell_dim], dtype=np.float64)
    v_box = float(np.prod(box))
    length = box[axis]

    # Seed-density profile along the axis.
    t = np.linspace(0.0, 1.0, 2001)
    if profile == "grain_size":
        grain = grain_min + (grain_max - grain_min) * t  # linear grain size
        dens = 1.0 / ((4.0 / 3.0) * np.pi * (grain / 2.0) ** 3)
    elif profile == "density":
        d_lo, d_hi = _seed_density(grain_min), _seed_density(grain_max)
        dens = d_lo + (d_hi - d_lo) * t  # linear seed density
    else:
        raise ValueError("profile must be 'grain_size' or 'density'")
    y_grid = t * length

    integral = float(np.sum(0.5 * (dens[:-1] + dens[1:]) * np.diff(y_grid)))
    n_seeds = int(round((v_box / length) * integral))
    n_seeds = max(1, min(n_seeds, max_seeds))

    # Rejection-sample the axis coordinate from the density profile; the
    # other two axes are uniform.
    rng = np.random.default_rng(rng_seed)
    d_max = float(dens.max())
    coords = np.empty(0, dtype=np.float64)
    while coords.size < n_seeds:
        cand = rng.uniform(0.0, length, n_seeds)
        keep = rng.uniform(0.0, 1.0, n_seeds) < np.interp(cand, y_grid, dens) / d_max
        coords = np.concatenate([coords, cand[keep]])
    seeds = rng.uniform(0.0, 1.0, (n_seeds, 3)) * box
    seeds[:, axis] = coords[:n_seeds]

    if weights is None:
        weights = dict(
            bond_weight=2.0, angle_weight=1.0, repulsion_weight=2.0,
            hard_core_scale=0.93, nonbond_push_scale=0.8,
        )

    cell = tc.Supercell.from_atoms(
        reference,
        cell_dim_angstroms=tuple(cell_dim),
        r_max=r_max, r_step=r_step, phi_num_bins=phi_num_bins,
        rng_seed=rng_seed,
    )
    # grain_size is a representative placeholder for bookkeeping only;
    # ``seeds`` overrides the actual placement.
    cell.generate(
        shell, num_steps=num_steps, show_progress=show_progress,
        grain_size=grain_max, seeds=seeds,
        displacement_sigma=displacement_sigma, **weights,
    )
    return cell.atoms
