"""Sliding-window (input, target) training-pair generation.

Slide a fixed window over a (periodic) blurred-potential stack and, for
each ``(cx, cy, z0)`` position, pair the **input** (the windowed
potential slice, in radians) with the **target** (the window-weighted
g2 / g3 of the true atoms over the same window).  Windows wrap across the
periodic faces, so a 50 Å in-plane cell stepped every 10 Å yields a full
5 × 5 grid, and several ``z`` cuts multiply that.

Window positions are independent, so generation parallelises across
processes (``n_jobs``); the cached ideal-gas envelope is built once in
the parent and seeded into the workers.
"""

from __future__ import annotations

import concurrent.futures as _cf
import multiprocessing as _mp
import os

import numpy as np

from .correlations import local_correlations
from .weighting import WindowSpec, windowed_image

__all__ = ["TrainingPair", "sliding_window_pairs"]

# Per-worker shared state (set once via the pool initializer).
_WORKER: dict = {}


class TrainingPair(dict):
    """A single training pair (a dict with attribute access for convenience).

    Keys: ``cx, cy, z0`` (window centre, Å), ``angle`` (input rotation,
    deg), ``input`` (2D windowed potential slice, rad), ``g2`` (1D),
    ``g3_slice`` (2D, (phi, r)), ``r`` (radial centres), ``phi_deg``
    (angle centres).
    """

    __getattr__ = dict.__getitem__


def _init_worker(potential_blurred, atoms, kw, cache_seed) -> None:
    """Pool initializer: stash shared state + seed the ideal-gas cache."""
    _WORKER["pb"] = potential_blurred
    _WORKER["atoms"] = atoms
    _WORKER["kw"] = kw
    if cache_seed:
        from .correlations import _RANDOM_CACHE

        _RANDOM_CACHE.update(cache_seed)


def _compute_pair(task):
    cx, cy, z0 = task
    pb, atoms, kw = _WORKER["pb"], _WORKER["atoms"], _WORKER["kw"]
    # The circular window makes g2 / g3 rotation-invariant, so compute the
    # target once and pair it with each rotated input image.
    corr = local_correlations(
        atoms,
        WindowSpec((float(cx), float(cy)), kw["side"], float(z0), kw["sigma_z"]),
        r_max=kw["r_max"],
        r_step=kw["r_step"],
        phi_num_bins=kw["phi_num_bins"],
        pair_peak=kw["pair_peak"],
        atom_scale=kw["atom_scale"],
        n_random=kw["n_random"],
        rng_seed=kw["rng_seed"],
    )
    arr = pb.array[pb.slice_index_for_z(z0)]  # (nx, ny)
    return [
        TrainingPair(
            cx=float(cx), cy=float(cy), z0=float(z0), angle=float(ang),
            input=windowed_image(arr, pb.sampling, (cx, cy), kw["side"], angle_deg=ang),
            g2=corr.g2, g3_slice=corr.g3_slice, r=corr.r, phi_deg=corr.phi_deg,
        )
        for ang in kw["rotations"]
    ]


def sliding_window_pairs(
    potential_blurred,
    atoms,
    *,
    pair_peak: float,
    r_max: float = 10.0,
    r_step: float = 0.1,
    phi_num_bins: int = 36,
    side: float | None = None,
    sigma_z: float | None = None,
    xy_step: float = 10.0,
    x_positions=None,
    y_positions=None,
    z_positions=None,
    rotations=None,
    atom_scale: np.ndarray | None = None,
    scattering_weighted: bool = True,
    rng_seed: int = 0,
    n_jobs: int = 1,
    show_progress: bool = True,
):
    """Generate (input, target) pairs over a sliding, wrapping window grid.

    Parameters
    ----------
    potential_blurred
        A blurred :class:`~tricor.ptycho.potential.PotentialStack`.
    atoms
        The structure the stack was built from.
    pair_peak
        Nearest-neighbour bond (Å) for the g3 r01 band.
    r_max, r_step, phi_num_bins
        Correlation binning.
    side
        Window side (Å); defaults to ``2 * r_max``.
    sigma_z
        Depth weight (Å); defaults to the stack's applied depth blur.
    xy_step
        In-plane grid step (Å).  A 50 Å cell with ``xy_step=10`` gives a
        5 × 5 grid (the cell is periodic, so the right/top edge is not
        repeated).
    x_positions, y_positions
        Explicit in-plane centres (Å) to sample, overriding ``xy_step``
        for that axis.  E.g. for a cell graded along y, pass
        ``x_positions=[Lx/2]`` and a fine ``y_positions`` to walk the
        transition.
    z_positions
        Iterable of ``z0`` depths (Å); defaults to four evenly spaced
        interior cuts.
    rotations
        Iterable of input-image rotations (degrees) about each window
        centre.  The circular window makes g2 / g3 rotation-invariant, so
        every angle reuses the same (once-computed) target and only the
        input is re-sampled (periodic bicubic) — free rotation
        augmentation.  Defaults to ``(0.0,)`` (no augmentation); pass e.g.
        ``np.arange(0, 360, 45)`` to emit 8 inputs per window.
    atom_scale, scattering_weighted
        Per-atom scattering weighting (computed once if not supplied).
    rng_seed
        Seed for the ideal-gas catalogue.
    n_jobs
        Number of worker processes.  ``1`` runs serially; ``-1`` uses all
        CPUs.  Windows are independent, so this scales near-linearly; the
        cached ideal-gas envelope is built once and shared with workers.
    show_progress
        Print a running count.

    Returns
    -------
    list[TrainingPair]
    """
    lx, ly = potential_blurred.extent
    lz = potential_blurred.n_slices * potential_blurred.slice_thickness
    if side is None:
        side = 2.0 * r_max
    if sigma_z is None:
        sigma_z = potential_blurred.blur[1] if potential_blurred.blur else 15.0
    if z_positions is None:
        z_positions = np.linspace(0, lz, 6)[1:-1]  # 4 interior cuts
    rotations = (0.0,) if rotations is None else tuple(float(a) for a in np.atleast_1d(rotations))
    if atom_scale is None and scattering_weighted:
        from .potential import scattering_power

        try:
            atom_scale = scattering_power(atoms.numbers)
        except Exception:  # noqa: BLE001
            atom_scale = None

    # Fix n_random so every window shares one cache key, and pre-build the
    # ideal-gas envelope once (seeded into workers, so it isn't rebuilt
    # per process).
    num_r = int(round(r_max / r_step))
    z_support = WindowSpec((0.0, 0.0), side, 0.0, sigma_z).z_support
    v_box = side * side * 2.0 * z_support
    n_random = int(np.clip(round(0.10 * v_box), 1500, 5000))
    from .correlations import _RANDOM_CACHE, _random_histograms

    # Build the ideal-gas envelope once in the parent and seed it into the
    # workers so it is not rebuilt per process.
    _random_histograms(WindowSpec((0.0, 0.0), side, 0.0, sigma_z),
                       num_r, r_step, phi_num_bins, n_random, rng_seed)
    cache_seed = dict(_RANDOM_CACHE)

    kw = dict(
        side=float(side), sigma_z=float(sigma_z), r_max=r_max, r_step=r_step,
        phi_num_bins=phi_num_bins, pair_peak=pair_peak, atom_scale=atom_scale,
        n_random=n_random, rng_seed=rng_seed, rotations=rotations,
    )

    xs = (np.asarray(x_positions, dtype=np.float64)
          if x_positions is not None else np.arange(0.0, lx - 1e-6, xy_step))
    ys = (np.asarray(y_positions, dtype=np.float64)
          if y_positions is not None else np.arange(0.0, ly - 1e-6, xy_step))
    tasks = [
        (float(cx), float(cy), float(z0))
        for z0 in np.atleast_1d(z_positions)
        for cx in xs
        for cy in ys
    ]
    total = len(tasks)

    if n_jobs == 1:
        _init_worker(potential_blurred, atoms, kw, None)  # cache already warm
        pairs = []
        for n, task in enumerate(tasks, 1):
            pairs.extend(_compute_pair(task))
            if show_progress:
                print(f"\r  windows: {n}/{total}  ({len(pairs)} pairs)", end="", flush=True)
        if show_progress:
            print()
        return pairs

    # Spawn (macOS-safe; fork crashes after Accelerate/BLAS is live).  The
    # worker functions live in this installed module, so this works in a
    # notebook; in a plain script guard the call with ``if __name__ ==
    # '__main__'``.  Shared state is pickled once per worker via the
    # initializer, not per task.
    workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    workers = max(1, min(workers, total))
    ctx = _mp.get_context("spawn")
    pairs = []
    with _cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(potential_blurred, atoms, kw, cache_seed),
    ) as ex:
        for n, pair_group in enumerate(ex.map(_compute_pair, tasks, chunksize=1), 1):
            pairs.extend(pair_group)
            if show_progress:
                print(f"\r  windows: {n}/{total}  ({len(pairs)} pairs, {workers} workers)", end="", flush=True)
    if show_progress:
        print()
    return pairs
