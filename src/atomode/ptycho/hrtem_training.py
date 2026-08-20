"""Sliding-window (input, target) HRTEM training-pair generation.

For each thickness ``t`` the exit wave has propagated through the block
``z ∈ [0, t]`` (abTEM enters at ``z = 0``).  We slide a circular window
over the HRTEM frame and pair the **input** (the windowed image — real
space, its diffractogram, and/or radial averages) with the **target**
(the window-weighted g2 / g3 of the true atoms in that depth block).

The target depends only on ``(thickness, window centre)`` — not on
defocus or in-plane rotation — so it is computed **once** per window and
reused across the whole defocus / rotation sweep.  Targets are the
expensive part (they need only the atoms), so they parallelise across
processes; the cheap HRTEM frames (a few FFTs from the cached exit waves)
are formed once per ``(thickness, defocus)`` in the parent.
"""

from __future__ import annotations

import concurrent.futures as _cf
import multiprocessing as _mp
import os

import numpy as np

from .correlations import local_correlations
from .hrtem import default_defocus, hrtem_image, hrtem_input
from .training import TrainingPair
from .weighting import WindowSpec, windowed_image

__all__ = ["sliding_window_pairs_hrtem"]

_REPRESENTATIONS = ("real", "fft", "real_radial", "fft_radial")
_WORKER: dict = {}


def _block_n_random(side: float, t: float) -> int:
    """Catalogue size for a block window (matches ``local_correlations``)."""
    return int(np.clip(round(0.10 * side * side * t), 1500, 5000))


def _init_worker(atoms, kw, cache_seed) -> None:
    _WORKER["atoms"] = atoms
    _WORKER["kw"] = kw
    if cache_seed:
        from .correlations import _RANDOM_CACHE

        _RANDOM_CACHE.update(cache_seed)


def _compute_target(task):
    """Block-window g2 / g3 target for one (thickness, window centre)."""
    ti, t, cx, cy = task
    atoms, kw = _WORKER["atoms"], _WORKER["kw"]
    corr = local_correlations(
        atoms,
        WindowSpec.block((float(cx), float(cy)), kw["side"], 0.0, float(t)),
        r_max=kw["r_max"],
        r_step=kw["r_step"],
        phi_num_bins=kw["phi_num_bins"],
        pair_peak=kw["pair_peak"],
        atom_scale=kw["atom_scale"],
        n_random=_block_n_random(kw["side"], float(t)),
        rng_seed=kw["rng_seed"],
    )
    return (int(ti), float(cx), float(cy), corr.g2, corr.g3_slice, corr.r, corr.phi_deg,
            int(corr.n_window))


def sliding_window_pairs_hrtem(
    stack,
    atoms,
    *,
    pair_peak: float,
    r_max: float = 10.0,
    r_step: float = 0.1,
    phi_num_bins: int = 36,
    side: float | None = None,
    thicknesses=None,
    defocus_offsets=(0.0,),
    semiangle_cutoff: float | None = None,
    cs: float = 0.0,
    xy_step: float = 10.0,
    x_positions=None,
    y_positions=None,
    rotations=None,
    representations=("real",),
    polar: dict | None = None,
    store_image: bool = True,
    atom_scale: np.ndarray | None = None,
    scattering_weighted: bool = True,
    rng_seed: int = 0,
    n_jobs: int = 1,
    show_progress: bool = True,
):
    """Generate (input, target) HRTEM pairs over a sliding window grid.

    Parameters
    ----------
    stack
        An :class:`~atomode.ptycho.hrtem.ExitWaveStack` (one multislice pass).
    atoms
        The structure the stack was propagated through.
    pair_peak
        Nearest-neighbour bond (Å) for the g3 r01 band.
    r_max, r_step, phi_num_bins
        Correlation binning.
    side
        Circular-window diameter (Å); defaults to ``2 * r_max``.
    thicknesses
        Which stack thicknesses (Å) to use; defaults to all of them.  Each
        maps to the depth block ``[0, t]`` for the target.
    defocus_offsets
        Offsets (Å) around the per-thickness centre defocus
        ``+t / 2`` (the mid-block focus).  E.g. ``(-60, -30, 0, 30, 60)``.
    semiangle_cutoff, cs
        Objective aperture (mrad) and spherical aberration (Å) for the lens.
    xy_step, x_positions, y_positions
        In-plane window grid (Å), exactly as the ptycho sampler.
    rotations
        Input-image rotations (deg); the circular window makes the target
        rotation-invariant, so each angle reuses the same target.  Defaults
        to ``(0.0,)``.
    representations
        Subset of ``("real", "fft", "real_radial", "fft_radial")`` to store
        on each pair.
    polar
        When set, a dict of keyword arguments for
        :func:`atomode.ptycho.polar.polar_fft_features`; each pair then
        carries a ``polar`` key holding the ``(max_order + 1, n_r)``
        angular-symmetry descriptor, built directly from the HRTEM frame.
        It is rotation invariant, so it is computed once per window and
        frame and shared across ``rotations``.
    store_image
        Keep the Cartesian windowed crop as ``input``.  Set ``False`` when
        only the ``polar`` descriptor is wanted, to skip the crop entirely.
    atom_scale, scattering_weighted
        Per-atom scattering weighting (computed once if not supplied).
    n_jobs
        Worker processes for the (expensive) target correlations.

    Returns
    -------
    list[TrainingPair]
        Each with ``cx, cy, thickness, defocus, angle``, ``input`` (real),
        any requested extra representations, and ``g2 / g3_slice / r /
        phi_deg``.
    """
    lx, ly = stack.extent
    if side is None:
        side = 2.0 * r_max
    if thicknesses is None:
        ti_list = list(range(stack.n_thicknesses))
    else:
        ti_list = [stack.thickness_index(t) for t in np.atleast_1d(thicknesses)]
    rotations = (0.0,) if rotations is None else tuple(float(a) for a in np.atleast_1d(rotations))
    defocus_offsets = tuple(float(o) for o in np.atleast_1d(defocus_offsets))
    reps = tuple(r for r in representations if r in _REPRESENTATIONS and r != "real")

    if atom_scale is None and scattering_weighted:
        from .potential import scattering_power

        try:
            atom_scale = scattering_power(atoms.numbers)
        except Exception:  # noqa: BLE001
            atom_scale = None

    xs = (np.asarray(x_positions, dtype=np.float64)
          if x_positions is not None else np.arange(0.0, lx - 1e-6, xy_step))
    ys = (np.asarray(y_positions, dtype=np.float64)
          if y_positions is not None else np.arange(0.0, ly - 1e-6, xy_step))

    num_r = int(round(r_max / r_step))
    kw = dict(
        side=float(side), r_max=r_max, r_step=r_step, phi_num_bins=phi_num_bins,
        pair_peak=pair_peak, atom_scale=atom_scale, rng_seed=rng_seed,
    )

    # Pre-build the ideal-gas envelope for each thickness block (cached and
    # seeded into the workers so it is not rebuilt per process).
    from .correlations import _RANDOM_CACHE, _random_histograms

    for ti in ti_list:
        t = float(stack.thicknesses[ti])
        _random_histograms(WindowSpec.block((0.0, 0.0), side, 0.0, t),
                           num_r, r_step, phi_num_bins, _block_n_random(side, t), rng_seed)
    cache_seed = dict(_RANDOM_CACHE)

    target_tasks = [
        (ti, float(stack.thicknesses[ti]), float(cx), float(cy))
        for ti in ti_list for cx in xs for cy in ys
    ]

    # --- targets (expensive, parallel; need only the atoms) ---
    target_map: dict[tuple[int, float, float], tuple] = {}

    def _store(res):
        ti, cx, cy, g2, g3, r, phi, nwin = res
        target_map[(ti, round(cx, 6), round(cy, 6))] = (g2, g3, r, phi, nwin)

    if n_jobs == 1:
        _init_worker(atoms, kw, None)  # cache already warm in-process
        for n, task in enumerate(target_tasks, 1):
            _store(_compute_target(task))
            if show_progress:
                print(f"\r  targets: {n}/{len(target_tasks)}", end="", flush=True)
        if show_progress:
            print()
    else:
        workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
        workers = max(1, min(workers, len(target_tasks)))
        ctx = _mp.get_context("spawn")
        with _cf.ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
            initializer=_init_worker, initargs=(atoms, kw, cache_seed),
        ) as ex:
            for n, res in enumerate(ex.map(_compute_target, target_tasks, chunksize=1), 1):
                _store(res)
                if show_progress:
                    print(f"\r  targets: {n}/{len(target_tasks)}  ({workers} workers)", end="", flush=True)
        if show_progress:
            print()

    # --- inputs (cheap, serial): one HRTEM frame per (thickness, defocus),
    # cropped at every window / rotation and paired with the stored target. ---
    pairs = []
    for ti in ti_list:
        t = float(stack.thicknesses[ti])
        center_df = default_defocus(t)
        for off in defocus_offsets:
            df = center_df + off
            full = hrtem_image(stack, ti, df, semiangle_cutoff=semiangle_cutoff, cs=cs)
            for cx in xs:
                for cy in ys:
                    g2, g3, r, phi, nwin = target_map[(ti, round(float(cx), 6), round(float(cy), 6))]
                    # Rotation invariant -> built once per window / frame.
                    polar_feat = None
                    if polar is not None:
                        from .polar import polar_fft_features

                        polar_feat = polar_fft_features(
                            full, stack.sampling, (float(cx), float(cy)), **polar
                        )
                        from .polar import default_orders

                        polar_ord = list(default_orders(polar.get("mode", "autocorrelation")))
                    for ang in rotations:
                        pair = TrainingPair(
                            cx=float(cx), cy=float(cy), thickness=t, defocus=float(df),
                            angle=float(ang), g2=g2, g3_slice=g3, r=r, phi_deg=phi,
                            n_window=int(nwin),
                        )
                        if store_image or reps:
                            im = windowed_image(full, stack.sampling,
                                                (float(cx), float(cy)), side, angle_deg=ang)
                            if store_image:
                                pair["input"] = im
                            if reps:
                                rep = hrtem_input(im)
                                for key in reps:
                                    pair[key] = rep[key]
                        if polar_feat is not None:
                            pair["polar"] = polar_feat
                            pair["polar_orders"] = polar_ord
                        pairs.append(pair)
        if show_progress:
            print(f"\r  frames: thickness {t:.0f} A done", end="", flush=True)
    if show_progress:
        print(f"\r  {len(pairs)} pairs from {len(ti_list)} thicknesses × "
              f"{len(defocus_offsets)} defocus × {len(xs)*len(ys)} windows × {len(rotations)} rot")
    return pairs
