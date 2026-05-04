"""Iterative grain-rotation refinement (Resampled Examples).

Given a supercell that has already been built by
:meth:`Supercell.generate` with grains, ``refine_grains`` walks the
grain list round-robin and, for each grain, tries a basket of
``(rotation, translation)`` pairs.  Each trial:

  * re-tiles the grain's Voronoi cell with the reference crystal
    rotated by ``R`` and offset by ``T``,
  * keeps the ``target_n`` nearest-to-seed atoms (so the grain's
    atom count stays constant),
  * runs a short FIRE relaxation on the grain plus its 1–2 nearest
    neighbour shells (the rest of the cell is frozen),
  * compares the post-FIRE total cost to the pre-trial cost.

The lowest-cost trial wins and the grain is updated in place.  The
loop continues round-robin until either no grain accepts a swap on
a full pass (converged) or a wall-clock budget is exhausted.

Why this works where atom-level thermal MC didn't: a grain rotation
is a *single coherent move* of ~50 atoms, large enough to reshape
local topology, while the local FIRE absorbs the boundary strain.
The Voronoi cell is constant (only the orientation of the lattice
inside it changes), so the global atom count and grain layout stay
the same — only the orientations of individual grains evolve.

Implementation notes
--------------------
* The Voronoi state (cells, master atom block, box dim) was saved on
  ``self`` by ``_build_grain_atoms``.  ``refine_grains`` re-uses it
  directly — the global Voronoi tessellation is *not* recomputed
  during refinement.
* Atom-count drift (different rotations of a non-spherical Voronoi
  cell hold slightly different numbers of master-block atoms) is
  resolved by always keeping the ``target_n`` nearest-to-seed
  candidates after the rotation and translation.  ``target_n`` is
  fixed to the grain's atom count at the start of refinement.
* Local cost vs. global cost: the bond/angle/repulsion cost outside
  the active grain's neighbourhood does not change during a trial
  (those atoms are frozen), so global ΔE equals local ΔE for the
  purposes of ranking trials.  We compare global costs.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .shells import CoordinationShellTarget
    from .supercell import Supercell

from ._grain import _periodic_voronoi_3d, _points_in_cell
from ._thermal_mc import _build_thermal_topology, _total_energy_fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _so3_random_rotation(
    rng: np.random.Generator,
    *,
    angle_min_rad: float = 0.18,   # ~10°
    angle_max_rad: float = np.pi,
) -> np.ndarray:
    """Sample a random rotation matrix with rotation angle in
    ``[angle_min, angle_max]``.  Axis is uniform on the sphere."""
    axis = rng.standard_normal(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    angle = float(rng.uniform(angle_min_rad, angle_max_rad))
    return _so3_axis_angle(axis, angle)


def _so3_bounded_rotation(
    rng: np.random.Generator,
    max_angle_rad: float,
) -> np.ndarray:
    """Sample a random rotation matrix with rotation angle uniform in
    ``[0, max_angle_rad]``.  Axis is uniform on the sphere.  Used by
    the coarse-to-fine refiner where ``max_angle_rad`` shrinks across
    amplitude phases."""
    axis = rng.standard_normal(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    angle = float(rng.uniform(0.0, max_angle_rad))
    return _so3_axis_angle(axis, angle)


def _so3_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit ``axis`` and ``angle``."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _retile_grain(
    *,
    master_positions: np.ndarray,
    master_numbers: np.ndarray,
    voronoi_cell: dict,
    seed_world: np.ndarray,
    box_dim: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    target_n: int,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Re-fill a single Voronoi cell with rotated + translated master
    atoms.  Returns ``(positions_world, numbers)`` or ``None`` when
    fewer than ``target_n`` candidates fall inside the cell.

    ``master_positions`` is in seed-local coordinates (the grain seed
    is at the origin of the master block).  We rotate and translate
    in seed-local coordinates, filter against the Voronoi convex
    hull, then shift to world coordinates and wrap into ``[0, L)``.
    """
    # Rotate in seed-local coords + apply lattice-anchor translation.
    candidates = master_positions @ rotation.T + translation[None, :]

    # Filter to atoms inside the Voronoi cell.
    inside = _points_in_cell(candidates, voronoi_cell, tol=1e-8)
    if not np.any(inside):
        return None
    cands_local = candidates[inside]
    cands_numbers = master_numbers[inside]

    if len(cands_local) < target_n:
        return None

    # Trim to ``target_n`` nearest atoms (in seed-local distance).
    distances_sq = np.sum(cands_local * cands_local, axis=1)
    order = np.argsort(distances_sq, kind="stable")[:target_n]
    kept_local = cands_local[order]
    kept_numbers = cands_numbers[order]

    # Shift to world coordinates and wrap.
    kept_world = (kept_local + seed_world) % box_dim
    return kept_world, kept_numbers


def _expand_neighborhood(
    *,
    grain_mask: np.ndarray,
    positions: np.ndarray,
    cell_mat: np.ndarray,
    cell_inv: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return a boolean mask: True for atoms in ``grain_mask`` plus
    any atoms within ``radius`` of any grain atom (PBC min-image)."""
    if not np.any(grain_mask):
        return np.zeros(len(positions), dtype=bool)
    grain_pos = positions[grain_mask]
    delta = positions[None, :, :] - grain_pos[:, None, :]
    frac = delta @ cell_inv
    frac -= np.rint(frac)
    delta_w = frac @ cell_mat
    d2 = np.sum(delta_w * delta_w, axis=2)  # (n_grain, N)
    near = (d2.min(axis=0) <= radius * radius)
    return near | grain_mask


def _pair_distance_cost(
    positions: np.ndarray,
    species_idx: np.ndarray,
    cell_mat: np.ndarray,
    cell_inv: np.ndarray,
    pair_peak: np.ndarray,           # (num_species, num_species)
    pair_hard_min: np.ndarray,       # (num_species, num_species)
    cutoff: float,
    *,
    grain_mask: "np.ndarray | None" = None,
    neighborhood_mask: "np.ndarray | None" = None,
) -> float:
    """Topology-free, position-only first-shell pair-distance cost.

    For each origin atom i (in grain_mask), score its first-shell
    neighbour environment using two terms:

      1. **Coordination reward** — for each j within the species-pair
         first-shell window ``[pair_inner, pair_outer]``, accumulate a
         triangular weight ``max(0, 1 - |d - peak| / margin)``.  The
         total approximates "fraction of expected coordination found"
         — higher is better.  We *negate* to convert to a cost.

      2. **Hard-core clash penalty** — quadratic if ``d <
         pair_hard_min``.

    Why this works across chemistry: the score uses
    ``shell_target.pair_inner`` / ``pair_outer`` to define the
    first-shell window per species pair, so we never reward 2nd-shell
    or further pairs.  This matters for FCC metals (Cu) and complex
    oxides (SiO₂, SrTiO₃) where multiple coordination shells exist;
    a naive ``(d - peak)²`` over an oversized cutoff would over-
    penalise legitimate 2nd-shell pairs and reward configurations
    with *fewer* in-cutoff neighbours.

    Position-only (no cached bond list), so the score is consistent
    across grains and trials.  Used by
    ``Supercell.refine_initial_orientations`` for sub-millisecond
    per-trial scoring.
    """
    if grain_mask is None:
        origins_idx = np.arange(positions.shape[0])
    else:
        origins_idx = np.flatnonzero(grain_mask)
    if neighborhood_mask is None:
        targets_idx = np.arange(positions.shape[0])
    else:
        targets_idx = np.flatnonzero(neighborhood_mask)

    if origins_idx.size == 0 or targets_idx.size == 0:
        return 0.0

    org_pos = positions[origins_idx]
    tgt_pos = positions[targets_idx]
    delta = tgt_pos[None, :, :] - org_pos[:, None, :]
    frac = delta @ cell_inv
    frac -= np.rint(frac)
    delta_w = frac @ cell_mat
    d2 = np.sum(delta_w * delta_w, axis=2)  # (n_org, n_tgt)
    d = np.sqrt(np.maximum(d2, 1e-30))

    # Pair-validity: not self, within scalar cutoff.
    self_mask = origins_idx[:, None] == targets_idx[None, :]
    in_cutoff = (d < cutoff) & (~self_mask)

    # Per-pair targets
    s_org = species_idx[origins_idx][:, None]
    s_tgt = species_idx[targets_idx][None, :]
    target_d = pair_peak[s_org, s_tgt]
    hard_d = pair_hard_min[s_org, s_tgt]

    # Bond-error: ``(d - target)²`` over pairs in cutoff.  Simple,
    # uniform across species pairs — empirically gives the cleanest
    # signal on Si and SiO₂.  (Relative-error scaling
    # ``(d - peak)² / peak²`` was tested and did not improve SiO₂.)
    bond_err = np.where(in_cutoff, (d - target_d) ** 2, 0.0)

    # Hard-core clash penalty over valid (non-self) pairs in cutoff.
    clash_excess = np.where(
        in_cutoff & (d < hard_d), hard_d - d, 0.0,
    )
    clash_penalty = (clash_excess ** 2) * 50.0

    cost = float(np.sum(bond_err) + np.sum(clash_penalty))
    return cost / max(origins_idx.size, 1)


def _global_cost(cell, shell_target, weights) -> float:
    """Total bond+angle+rep cost per atom of cell.atoms via the numba
    kernel (O(N·K), fast)."""
    species_idx = (
        cell._atom_shell_species_index
        if getattr(cell, "_atom_shell_species_index", None) is not None
        else cell._atom_species_index
    ).astype(np.intp, copy=True)
    topo = _build_thermal_topology(
        cell.atoms, species_idx, shell_target,
        hard_core_scale=float(weights.get("hard_core_scale", 1.0)),
        nonbond_push_scale=float(weights.get("nonbond_push_scale", 1.0)),
    )
    cell_mat = np.ascontiguousarray(cell.atoms.cell.array, dtype=np.float64)
    cell_inv = np.linalg.inv(cell_mat)
    positions = np.ascontiguousarray(cell.atoms.positions, dtype=np.float64)
    num_atoms = positions.shape[0]
    r_dummy = np.zeros_like(positions)
    total, eb, ea, er = _total_energy_fast(
        positions, species_idx, cell_mat, cell_inv,
        topo["bond_i"], topo["bond_j"], topo["bond_r_target"],
        float(weights.get("bond_weight", 1.0)),
        topo["tri_center"], topo["tri_a"], topo["tri_b"],
        topo["tri_phi_target"],
        float(weights.get("angle_weight", 0.5)),
        topo["rep_atom_start"], topo["rep_atom_list"],
        topo["hard_core"], topo["nonbond_push"],
        float(weights.get("repulsion_weight", 3.0)),
        topo["bonded_flat"], num_atoms,
        r_dummy, 0.0,
    )
    n = max(1, num_atoms)
    return dict(
        total=float(total) / n,
        bond=float(eb) / n,
        angle=float(ea) / n,
        rep=float(er) / n,
    )


# ---------------------------------------------------------------------------
# Main mixin
# ---------------------------------------------------------------------------


class _ResampleMixin:
    """Adds :meth:`refine_grains` to ``Supercell``."""

    def refine_grains(
        self: "Supercell",
        shell_target: "CoordinationShellTarget",
        *,
        # Time + iteration budget
        time_budget_sec: float = 300.0,
        max_passes: int = 20,
        # Per-grain trial budget
        n_rot: int = 8,
        n_trans: int = 4,
        rotation_min_deg: float = 10.0,
        rotation_max_deg: float = 180.0,
        # Local FIRE
        local_fire_steps: int = 50,
        neighbor_shell_radius_factor: float = 2.5,
        # Spring weights (mirror generate())
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 3.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        # Final FIRE quench (no restraint, no freeze) to settle the
        # whole cell into a clean local minimum.
        final_quench_steps: int = 200,
        # Output
        capture_trajectory: bool = True,
        show_progress: bool = True,
        rng_seed: "int | None" = None,
    ) -> dict:
        """Iterative per-grain rotation refinement with local FIRE.

        Walks the grain list round-robin; for each grain, tries
        ``n_rot * n_trans`` candidate (rotation, translation) pairs,
        keeping the lowest-cost outcome.  A trial is judged by total
        bond + angle + repulsion cost after a short
        ``local_fire_steps``-step FIRE pass on the grain plus its
        neighbour shell.

        Returns the history dict that gets stashed on
        ``self.refine_grains_history`` and used by
        :meth:`Supercell.export_trajectory_html(history='refine_grains')`.

        Parameters
        ----------
        time_budget_sec
            Stop after this many seconds of wall-clock have elapsed.
            For demo-quality results 60 - 300 s is plenty; for
            production runs raise to 1800 - 3600 s.
        max_passes
            Stop after this many full round-robin passes regardless
            of time.  A pass with zero accepts triggers early
            convergence.
        n_rot : int, optional
            Per-grain rotation trials.  Wall time scales linearly
            with ``n_rot * n_trans * num_grains``.
        n_trans : int, optional
            Per-grain translation trials.  Combined with ``n_rot``
            sets the basin coverage per grain.
        rotation_min_deg : float, optional
            Lower bound on trial rotation angle (degrees).  Lower
            values encourage fine local refinements late in the
            search.
        rotation_max_deg : float, optional
            Upper bound on trial rotation angle (degrees).  Trial
            rotations sample uniformly in
            ``[rotation_min_deg, rotation_max_deg]`` about a
            uniformly-random axis.
        neighbor_shell_radius_factor
            The local FIRE region around each grain extends
            ``radius_factor * pair_peak_max`` Å beyond the grain
            atoms.  ~2.5 covers the grain's own atoms plus the
            atoms in adjacent grains that share a Voronoi face,
            which is what bonds across grain boundaries actually
            depend on.
        local_fire_steps
            Number of FIRE descent steps per trial.  50 is enough to
            absorb the boundary strain from a fresh rotation; raising
            past 100 wastes budget.
        bond_weight : float, optional
            Spring weight for bond-distance terms.  Mirror the value
            used when the cell was originally generated.
        angle_weight : float, optional
            Spring weight for bond-angle terms.
        repulsion_weight : float, optional
            Spring weight for hard-core + nonbond-clearance terms.
        hard_core_scale : float, optional
            Multiplier on ``shell_target.pair_inner`` setting the
            minimum allowed pair distance.
        nonbond_push_scale : float, optional
            Multiplier on ``shell_target.pair_peak`` setting the
            non-bonded shell-clearance radius.
        final_quench_steps
            After the round-robin loop finishes, run this many
            full-cell FIRE steps (no freezing, no restraint) to lock
            in the global minimum.  Set 0 to skip.
        capture_trajectory
            When True, append a frame to the history each time a
            grain rotation is accepted.  Adds the initial state and
            the post-final-quench state too.

        Notes
        -----
        Wall-clock for the demo runs in this repo
        (``cell_dim_angstroms = (40, 40, 40)``):

        ===========  =======  =============  ===========
        regime       grains   ~accepts/run   wall time
        ===========  =======  =============  ===========
        amorphous    ~600     ~50            5–7 min
        MRO          ~50      ~10            2–3 min
        nano (NC)    ~5       ~3             30–60 s
        ===========  =======  =============  ===========

        For production-quality output (e.g. ML training datasets),
        raise ``time_budget_sec`` to 1800 - 3600, raise ``n_rot`` to
        16 - 32, and run multiple seeds in parallel.
        """
        # --- prerequisites ---
        if getattr(self, "_grain_ids", None) is None:
            raise ValueError(
                "refine_grains requires a grain-built cell.  "
                "Call Supercell.generate(..., grain_size=...) first."
            )
        if getattr(self, "_grain_cells", None) is None:
            raise ValueError(
                "refine_grains requires Voronoi cells cached on the "
                "supercell.  Re-run generate() to populate them."
            )

        rng = (np.random.default_rng(rng_seed)
               if rng_seed is not None else self.rng)

        weights = dict(
            bond_weight=float(bond_weight),
            angle_weight=float(angle_weight),
            repulsion_weight=float(repulsion_weight),
            hard_core_scale=float(hard_core_scale),
            nonbond_push_scale=float(nonbond_push_scale),
        )

        # --- pull cached grain state ---
        grain_ids = np.asarray(self._grain_ids, dtype=np.intp)
        grain_seeds = np.asarray(self._grain_seeds, dtype=np.float64)
        voronoi_cells = self._grain_cells
        masters = self._grain_masters
        # Grain source assignment: for single-source builds this is
        # all zeros; for multi-source builds (carbon sp²/sp³) it
        # carries the source index per grain.
        grain_source = self._grain_source
        if grain_source is None:
            grain_source = np.zeros(len(grain_seeds), dtype=np.intp)
        is_crystalline = np.asarray(
            self._grain_is_crystalline, dtype=bool,
        )
        box_dim = np.asarray(self._grain_box_dim, dtype=np.float64)

        # Lattice translation range: random offset within the master
        # crystal's primitive cell, so we explore lattice anchor
        # positions for each rotation.
        master_lattice = np.asarray(
            self._grain_master_lattice, dtype=np.float64,
        )
        # Half the unit-cell side is enough; bigger offsets are
        # equivalent up to the lattice periodicity.
        translation_basis = master_lattice * 0.5

        # --- FIRE neighbourhood radius ---
        pair_peak_max = float(np.max(
            np.asarray(shell_target.pair_peak, dtype=np.float64)
        ))
        neighborhood_radius = float(
            neighbor_shell_radius_factor * pair_peak_max
        )

        cell_mat = np.ascontiguousarray(
            self.atoms.cell.array, dtype=np.float64,
        )
        cell_inv = np.linalg.inv(cell_mat)

        # --- Crystalline grains only (amorphous grains stay put) ---
        unique_grains = [
            int(g) for g in np.unique(grain_ids[grain_ids >= 0])
            if is_crystalline[int(g)]
        ]
        if not unique_grains:
            raise ValueError(
                "No crystalline grains found.  refine_grains has "
                "nothing to refine."
            )

        # When there are very few grains (e.g. NC with 2-4 grains in
        # a 20-40 Å cell), each grain's "neighbourhood" covers almost
        # the entire cell — so the per-grain warmup FIRE is
        # essentially whole-cell FIRE on the post-generate state, and
        # subsequent trials face a too-tight baseline that blocks
        # otherwise-good rotations from accepting.  In that low-count
        # regime we skip the warmup and compare trials directly
        # against the post-generate cost (still apples-to-apples
        # because all trials *and* the baseline are at the same FIRE
        # level — namely, the FIRE that was applied inside generate()
        # itself, with no extra constrained-FIRE pass on either side).
        do_warmup = len(unique_grains) > 4

        # --- History accumulator ---
        history: dict = dict(
            iteration=[],
            global_cost=[],
            cost_bond=[],
            cost_angle=[],
            cost_rep=[],
            accepted_grain=[],
            rotation_deg=[],
            translation_norm=[],
            pass_index=[],
            trajectory=[] if capture_trajectory else None,
        )

        rotation_min_rad = np.deg2rad(float(rotation_min_deg))
        rotation_max_rad = np.deg2rad(float(rotation_max_deg))

        def _record(it_idx, accepted_g, rot_deg, trans_norm, pass_idx):
            cost = _global_cost(self, shell_target, weights)
            history["iteration"].append(it_idx)
            history["global_cost"].append(cost["total"])
            history["cost_bond"].append(cost["bond"])
            history["cost_angle"].append(cost["angle"])
            history["cost_rep"].append(cost["rep"])
            history["accepted_grain"].append(accepted_g)
            history["rotation_deg"].append(rot_deg)
            history["translation_norm"].append(trans_norm)
            history["pass_index"].append(pass_idx)
            if capture_trajectory:
                history["trajectory"].append(
                    self.atoms.positions.copy().astype(np.float32)
                )

        # Initial frame at iteration 0.
        _record(0, -1, 0.0, 0.0, 0)

        # --- Round-robin loop ---
        t0 = time.time()
        iteration = 0
        for pass_idx in range(1, max_passes + 1):
            grain_order = list(unique_grains)
            rng.shuffle(grain_order)
            improved_this_pass = False

            for gid in grain_order:
                if time.time() - t0 >= time_budget_sec:
                    break

                grain_mask = (grain_ids == gid)
                target_n = int(np.sum(grain_mask))
                if target_n == 0:
                    continue

                # Seed-world for this grain (post any single-grain
                # shared-shift handling).
                seed_world = grain_seeds[gid]
                voronoi_cell = voronoi_cells[gid]
                src_idx = int(grain_source[gid])
                master = masters[src_idx]
                master_pos = np.asarray(master["positions"], dtype=np.float64)
                master_num = np.asarray(master["numbers"], dtype=np.int64)

                # Snapshot pre-trial state (positions only — species
                # mapping is preserved by atom indices).
                pre_positions = self.atoms.positions.copy()

                neighborhood_mask = _expand_neighborhood(
                    grain_mask=grain_mask,
                    positions=self.atoms.positions,
                    cell_mat=cell_mat, cell_inv=cell_inv,
                    radius=neighborhood_radius,
                )
                freeze_mask = ~neighborhood_mask

                # Apples-to-apples warmup (when enough grains exist).
                # generate()'s shell_relax ran on the *full cell* with
                # no freeze_mask, but trials only get FIRE on a
                # *constrained subspace* (atoms outside
                # ``neighborhood_mask`` frozen).  Running the same
                # ``local_fire_steps`` FIRE on the unmodified state
                # with the same freeze_mask brings ``pre_cost`` down
                # to the constrained-local-min surface that trials
                # land on, so the comparison is fair.
                #
                # Skipped when ``do_warmup`` is False (low grain count
                # cases — see the explanation above ``unique_grains``).
                if do_warmup:
                    self.shell_relax(
                        shell_target,
                        num_steps=int(local_fire_steps),
                        freeze_mask=freeze_mask,
                        neighbor_update_interval=99999,
                        capture_trajectory=False,
                        show_progress=False,
                        **weights,
                    )
                warmup_positions = self.atoms.positions.copy()
                pre_cost = _global_cost(self, shell_target, weights)["total"]

                best_cost = pre_cost
                best_positions: "np.ndarray | None" = None
                best_rot_deg = 0.0
                best_trans_norm = 0.0

                for r_trial in range(n_rot):
                    rotation = _so3_random_rotation(
                        rng,
                        angle_min_rad=rotation_min_rad,
                        angle_max_rad=rotation_max_rad,
                    )
                    rot_deg = np.rad2deg(np.arccos(np.clip(
                        (np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0,
                    )))
                    for t_trial in range(n_trans):
                        # Random offset within ±0.5 master-unit-cell
                        # along each lattice direction (lattice
                        # anchor search).
                        frac = rng.uniform(-1.0, 1.0, size=3)
                        translation = frac @ translation_basis
                        trans_norm = float(np.linalg.norm(translation))

                        retile = _retile_grain(
                            master_positions=master_pos,
                            master_numbers=master_num,
                            voronoi_cell=voronoi_cell,
                            seed_world=seed_world,
                            box_dim=box_dim,
                            rotation=rotation,
                            translation=translation,
                            target_n=target_n,
                        )
                        if retile is None:
                            # Not enough candidates — try next trial.
                            self.atoms.positions = warmup_positions.copy()
                            continue
                        new_positions_world, _new_numbers = retile

                        # Reset neighbourhood to the warmup state so
                        # every trial starts from the same baseline,
                        # then install the rotated grain on top.
                        self.atoms.positions = warmup_positions.copy()
                        self.atoms.positions[grain_mask] = new_positions_world

                        # Local FIRE: only the neighbourhood moves.
                        self.shell_relax(
                            shell_target,
                            num_steps=int(local_fire_steps),
                            freeze_mask=freeze_mask,
                            # One topology rebuild at the start of the
                            # local FIRE is enough — atoms move at
                            # most fractions of an Å in 30-50 FIRE
                            # steps, so the bond graph is stable.
                            neighbor_update_interval=99999,
                            capture_trajectory=False,
                            show_progress=False,
                            **weights,
                        )
                        # Rank trials by global bond+angle+rep cost
                        # (the same metric the kernel optimizes), not
                        # by shell_relax's ``final_loss`` which uses
                        # MEAN-bond-stretch and a count-based rep
                        # term, neither of which match what the
                        # downstream MC / shell_target consumers
                        # actually care about.
                        cost = _global_cost(
                            self, shell_target, weights,
                        )["total"]
                        if cost < best_cost:
                            best_cost = cost
                            best_positions = self.atoms.positions.copy()
                            best_rot_deg = float(rot_deg)
                            best_trans_norm = trans_norm

                        # Reset positions for the next trial in the
                        # same grain (back to the post-warmup baseline,
                        # not the pre-warmup state).
                        self.atoms.positions = warmup_positions.copy()

                # Apply the best trial; otherwise keep the
                # post-warmup state (it's strictly better than the
                # pre-warmup state since it had ``local_fire_steps``
                # of constrained gradient descent).
                if best_positions is not None:
                    self.atoms.positions = best_positions
                    iteration += 1
                    improved_this_pass = True
                    _record(iteration, int(gid),
                            best_rot_deg, best_trans_norm, pass_idx)
                    if show_progress:
                        elapsed = time.time() - t0
                        print(
                            f"  pass {pass_idx} grain {gid}: "
                            f"cost {pre_cost:.4f} -> {best_cost:.4f} "
                            f"(rot {best_rot_deg:.1f}°, "
                            f"trans {best_trans_norm:.2f} Å, "
                            f"elapsed {elapsed:.1f} s)"
                        )
                else:
                    self.atoms.positions = warmup_positions

            # End of pass.
            if not improved_this_pass:
                if show_progress:
                    print(f"  pass {pass_idx}: no improvements — converged")
                break
            if time.time() - t0 >= time_budget_sec:
                if show_progress:
                    print(f"  pass {pass_idx}: time budget exhausted")
                break

        # --- Final FIRE quench (whole cell, no restraint) ---
        if final_quench_steps > 0:
            if show_progress:
                print(f"  final FIRE quench ({final_quench_steps} steps)...")
            self.shell_relax(
                shell_target,
                num_steps=int(final_quench_steps),
                k_restraint=0.0,
                capture_trajectory=False,
                show_progress=False,
                **weights,
            )
            iteration += 1
            _record(iteration, -2, 0.0, 0.0, -1)  # -2 = final quench

        # --- Pack history into the trajectory-export-friendly form ---
        traj = (
            np.asarray(history["trajectory"], dtype=np.float32)
            if capture_trajectory else None
        )
        history_out = dict(
            iteration=np.asarray(history["iteration"], dtype=np.intp),
            global_cost=np.asarray(history["global_cost"], dtype=np.float64),
            cost_bond=np.asarray(history["cost_bond"], dtype=np.float64),
            cost_angle=np.asarray(history["cost_angle"], dtype=np.float64),
            cost_rep=np.asarray(history["cost_rep"], dtype=np.float64),
            accepted_grain=np.asarray(history["accepted_grain"], dtype=np.intp),
            rotation_deg=np.asarray(history["rotation_deg"], dtype=np.float64),
            translation_norm=np.asarray(
                history["translation_norm"], dtype=np.float64,
            ),
            pass_index=np.asarray(history["pass_index"], dtype=np.intp),
            trajectory=traj,
            # Aliases so export_trajectory_html(history='refine_grains')
            # finds what it needs.  The viewer only looks for
            # 'trajectory'; the rest is metadata for the cost plot.
        )

        self.refine_grains_history = history_out

        # Invalidate the g3 cache and spatial index so subsequent
        # measure_g3 / view_structure calls see the refined cell.
        self.current_distribution = None
        if hasattr(self, "_rebuild_spatial_index"):
            self._rebuild_spatial_index()

        return history_out

    # =====================================================================
    # Coarse-to-fine variant
    # =====================================================================

    def refine_grains_coarse_to_fine(
        self: "Supercell",
        shell_target: "CoordinationShellTarget",
        *,
        # Stage 1: uniform-SO(3) basin search.  Each trial samples a
        # rotation independently of the grain's current orientation —
        # this is what lets us *find* a basin different from
        # generate()'s starting point.
        initial_uniform_trials: int = 32,
        # Stage 2: bounded-amplitude refinement around the current
        # best orientation, coarse to fine.  Each amplitude bounds
        # ``R_delta`` and the trial composes ``R_current @ R_delta``.
        angle_schedule_deg: tuple = (45.0, 22.0, 11.0, 5.0, 2.0, 1.0),
        trials_per_amplitude: int = 12,
        max_rounds_per_amplitude: int = 3,
        # Local FIRE
        local_fire_steps: int = 50,
        neighbor_shell_radius_factor: float = 2.5,
        # Spring weights
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 3.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        # Final whole-cell FIRE quench
        final_quench_steps: int = 200,
        # Output
        time_budget_sec: float = 180.0,
        capture_trajectory: bool = True,
        show_progress: bool = True,
        rng_seed: "int | None" = None,
    ) -> dict:
        """Coarse-to-fine basin hopping for per-grain rotations.

        Walks an angle-amplitude schedule from coarse to fine.  For
        each ``(amplitude, grain)`` pair, samples
        ``trials_per_amplitude`` candidate rotations bounded in angle
        by ``amplitude`` and **composed** onto the grain's current
        orientation, applies a short local FIRE on each, and accepts
        the lowest-cost outcome.  The same amplitude is repeated up
        to ``max_rounds_per_amplitude`` rounds while it keeps producing
        accepts; then the schedule steps down.

        This finds dozens of accepts where uniform-SO(3) sampling
        finds only 1-2 because every trial is anchored to the current
        orientation, so fine refinements within the chosen basin keep
        yielding small improvements; translations are also bounded by
        the amplitude (``trans_scale = amplitude / 180°``); and
        multi-round at each amplitude exhausts the local search
        before stepping to a smaller amplitude.

        Parameters
        ----------
        shell_target : CoordinationShellTarget
            Target whose ``pair_peak`` defines the per-pair bond
            length the cost function targets.
        initial_uniform_trials : int, optional
            Stage-1 uniform-SO(3) basin search per grain (rotations
            sampled independently of the current orientation, used
            only to escape the seed orientation).  Default ``32``.
            Set ``0`` to skip the basin search.
        angle_schedule_deg : tuple of float, optional
            Stage-2 amplitude schedule (degrees), coarse → fine.
            Default ``(45, 22, 11, 5, 2, 1)``.
        trials_per_amplitude : int, optional
            Candidate rotations per (amplitude, grain).  Default
            ``12``.
        max_rounds_per_amplitude : int, optional
            Maximum round-robin passes over all grains within one
            amplitude phase.  Default ``3``.
        local_fire_steps : int, optional
            FIRE steps applied to each trial's neighbour shell to
            evaluate its cost.  Default ``50``.
        neighbor_shell_radius_factor : float, optional
            Local-FIRE neighbour shell extends to this multiple of
            ``shell_target.pair_peak`` around the perturbed grain.
            Default ``2.5``.
        bond_weight, angle_weight, repulsion_weight : float, optional
            Spring weights forwarded to the per-trial FIRE + cost
            evaluation.
        hard_core_scale, nonbond_push_scale : float, optional
            Repulsion thresholds for the per-trial FIRE.
        final_quench_steps : int, optional
            Whole-cell FIRE quench steps applied after the SO(3)
            search completes.  Default ``200``; set ``0`` to skip.
        time_budget_sec : float, optional
            Wall-time guard rail (seconds).  The search bails after
            this even if amplitudes remain.  Default ``180``.
        capture_trajectory : bool, optional
            Record per-frame atom positions after each acceptance.
            Default ``True`` (needed for trajectory-replay HTML).
        show_progress : bool, optional
            Display a tqdm progress bar.  Default ``True``.
        rng_seed : int, optional
            Seed for the rotation sampler.  ``None`` (default) uses
            the cell's own RNG.

        Returns
        -------
        dict
            History captured under
            ``self.refine_grains_coarse_to_fine_history`` — same
            shape as :meth:`refine_grains`'s history (so the
            trajectory + cost plotters work unchanged), plus a
            ``rotation_amplitude_deg`` array marking which amplitude
            phase produced each accept.
        """
        if getattr(self, "_grain_ids", None) is None:
            raise ValueError(
                "refine_grains_coarse_to_fine requires a grain-built "
                "cell.  Call Supercell.generate(grain_size=...) first.")
        if getattr(self, "_grain_cells", None) is None:
            raise ValueError(
                "refine_grains_coarse_to_fine requires Voronoi cells "
                "cached on the supercell.  Re-run generate() to "
                "populate them.")

        rng = (np.random.default_rng(rng_seed)
               if rng_seed is not None else self.rng)

        weights = dict(
            bond_weight=float(bond_weight),
            angle_weight=float(angle_weight),
            repulsion_weight=float(repulsion_weight),
            hard_core_scale=float(hard_core_scale),
            nonbond_push_scale=float(nonbond_push_scale),
        )

        grain_ids = np.asarray(self._grain_ids, dtype=np.intp)
        grain_seeds = np.asarray(self._grain_seeds, dtype=np.float64)
        voronoi_cells = self._grain_cells
        masters = self._grain_masters
        grain_source = self._grain_source
        if grain_source is None:
            grain_source = np.zeros(len(grain_seeds), dtype=np.intp)
        is_crystalline = np.asarray(
            self._grain_is_crystalline, dtype=bool,
        )
        box_dim = np.asarray(self._grain_box_dim, dtype=np.float64)
        master_lattice = np.asarray(
            self._grain_master_lattice, dtype=np.float64,
        )
        # Per-direction translation amplitude = half the lattice basis
        # vectors (anything bigger is symmetry-equivalent modulo the
        # lattice).
        translation_basis = master_lattice * 0.5

        pair_peak_max = float(np.max(
            np.asarray(shell_target.pair_peak, dtype=np.float64)
        ))
        neighborhood_radius = float(
            neighbor_shell_radius_factor * pair_peak_max
        )

        cell_mat = np.ascontiguousarray(
            self.atoms.cell.array, dtype=np.float64,
        )
        cell_inv = np.linalg.inv(cell_mat)

        unique_grains = [
            int(g) for g in np.unique(grain_ids[grain_ids >= 0])
            if is_crystalline[int(g)]
        ]
        if not unique_grains:
            raise ValueError(
                "No crystalline grains found.  "
                "refine_grains_coarse_to_fine has nothing to refine.")

        # Per-grain orientation state.  Initialised from the rotations
        # ``generate()`` applied; updates on each accepted trial via
        # ``R_current[g] = R_current[g] @ R_delta``.
        max_grain_id = int(grain_ids.max()) + 1
        current_rotations = np.zeros((max_grain_id, 3, 3), dtype=np.float64)
        for g in range(max_grain_id):
            current_rotations[g] = (
                self._grain_rotations_initial[g]
                if g < len(self._grain_rotations_initial)
                else np.eye(3)
            )
        current_translations = np.zeros((max_grain_id, 3), dtype=np.float64)

        history: dict = dict(
            iteration=[],
            global_cost=[],
            cost_bond=[],
            cost_angle=[],
            cost_rep=[],
            accepted_grain=[],
            rotation_amplitude_deg=[],
            amplitude_phase=[],
            trajectory=[] if capture_trajectory else None,
        )

        def _record(it_idx, accepted_g, amp_deg, phase_idx):
            cost = _global_cost(self, shell_target, weights)
            history["iteration"].append(it_idx)
            history["global_cost"].append(cost["total"])
            history["cost_bond"].append(cost["bond"])
            history["cost_angle"].append(cost["angle"])
            history["cost_rep"].append(cost["rep"])
            history["accepted_grain"].append(accepted_g)
            history["rotation_amplitude_deg"].append(amp_deg)
            history["amplitude_phase"].append(phase_idx)
            if capture_trajectory:
                history["trajectory"].append(
                    self.atoms.positions.copy().astype(np.float32)
                )

        _record(0, -1, 0.0, -1)

        iteration = 0
        t0 = time.time()

        # =================================================================
        # Stage 1: uniform-SO(3) basin search per grain.
        # Each trial proposes an independent random rotation drawn
        # uniformly from a wide angle range — *not* composed with
        # the grain's current orientation — so we can escape the
        # starting basin from generate() and land somewhere new.
        # =================================================================
        if initial_uniform_trials > 0:
            grain_order = list(unique_grains)
            rng.shuffle(grain_order)
            for gid in grain_order:
                if time.time() - t0 >= time_budget_sec:
                    break

                grain_mask = (grain_ids == gid)
                target_n = int(np.sum(grain_mask))
                if target_n == 0:
                    continue
                seed_world = grain_seeds[gid]
                voronoi_cell = voronoi_cells[gid]
                src_idx = int(grain_source[gid])
                master = masters[src_idx]
                master_pos = np.asarray(
                    master["positions"], dtype=np.float64,
                )
                master_num = np.asarray(
                    master["numbers"], dtype=np.int64,
                )

                snapshot = self.atoms.positions.copy()
                pre_cost = _global_cost(
                    self, shell_target, weights,
                )["total"]
                best_cost = pre_cost
                best_R = None
                best_T = None
                best_atoms: "np.ndarray | None" = None

                neighborhood_mask = _expand_neighborhood(
                    grain_mask=grain_mask,
                    positions=self.atoms.positions,
                    cell_mat=cell_mat, cell_inv=cell_inv,
                    radius=neighborhood_radius,
                )
                freeze_mask = ~neighborhood_mask

                for _trial in range(initial_uniform_trials):
                    # Direct uniform-SO(3) sample (NOT composed): the
                    # trial is independent of current_rotations[gid].
                    R_trial = _so3_random_rotation(
                        rng,
                        angle_min_rad=np.deg2rad(10.0),
                        angle_max_rad=np.pi,
                    )
                    frac = rng.uniform(-1.0, 1.0, size=3)
                    T_trial = frac @ translation_basis

                    retile = _retile_grain(
                        master_positions=master_pos,
                        master_numbers=master_num,
                        voronoi_cell=voronoi_cell,
                        seed_world=seed_world,
                        box_dim=box_dim,
                        rotation=R_trial,
                        translation=T_trial,
                        target_n=target_n,
                    )
                    if retile is None:
                        self.atoms.positions = snapshot.copy()
                        continue
                    new_pos_world, _ = retile

                    self.atoms.positions = snapshot.copy()
                    self.atoms.positions[grain_mask] = new_pos_world

                    self.shell_relax(
                        shell_target,
                        num_steps=int(local_fire_steps),
                        freeze_mask=freeze_mask,
                        neighbor_update_interval=99999,
                        capture_trajectory=False,
                        show_progress=False,
                        **weights,
                    )

                    cost = _global_cost(
                        self, shell_target, weights,
                    )["total"]
                    if cost < best_cost:
                        best_cost = cost
                        best_R = R_trial
                        best_T = T_trial
                        best_atoms = self.atoms.positions.copy()

                    self.atoms.positions = snapshot.copy()

                self.atoms.positions = snapshot.copy()
                if best_atoms is not None:
                    self.atoms.positions = best_atoms
                    current_rotations[gid] = best_R
                    current_translations[gid] = best_T
                    iteration += 1
                    _record(iteration, int(gid), 180.0, -2)
                    if show_progress:
                        elapsed = time.time() - t0
                        print(
                            f"  uniform   grain {gid}: "
                            f"cost {pre_cost:.4f} -> {best_cost:.4f} "
                            f"(elapsed {elapsed:.1f} s)"
                        )

        # =================================================================
        # Stage 2: bounded-amplitude refinement around the current
        # orientation.  Each trial composes ``R_current @ R_delta``
        # so we drill into the basin found in Stage 1.
        # =================================================================
        for phase_idx, amplitude_deg in enumerate(angle_schedule_deg):
            if time.time() - t0 >= time_budget_sec:
                break
            amp_rad = np.deg2rad(float(amplitude_deg))
            trans_scale = float(amplitude_deg / 180.0)

            for round_idx in range(max_rounds_per_amplitude):
                if time.time() - t0 >= time_budget_sec:
                    break
                grain_order = list(unique_grains)
                rng.shuffle(grain_order)
                improved_this_round = False

                for gid in grain_order:
                    if time.time() - t0 >= time_budget_sec:
                        break

                    grain_mask = (grain_ids == gid)
                    target_n = int(np.sum(grain_mask))
                    if target_n == 0:
                        continue
                    seed_world = grain_seeds[gid]
                    voronoi_cell = voronoi_cells[gid]
                    src_idx = int(grain_source[gid])
                    master = masters[src_idx]
                    master_pos = np.asarray(
                        master["positions"], dtype=np.float64,
                    )
                    master_num = np.asarray(
                        master["numbers"], dtype=np.int64,
                    )

                    snapshot = self.atoms.positions.copy()
                    pre_cost = _global_cost(
                        self, shell_target, weights,
                    )["total"]
                    best_cost = pre_cost
                    best_R = None
                    best_T = None
                    best_atoms: "np.ndarray | None" = None

                    neighborhood_mask = _expand_neighborhood(
                        grain_mask=grain_mask,
                        positions=self.atoms.positions,
                        cell_mat=cell_mat, cell_inv=cell_inv,
                        radius=neighborhood_radius,
                    )
                    freeze_mask = ~neighborhood_mask

                    for trial in range(trials_per_amplitude):
                        # Compose: R_trial = R_current @ R_delta with
                        # |R_delta| ≤ amplitude.  This anchors the trial
                        # to the grain's current orientation.
                        R_delta = _so3_bounded_rotation(rng, amp_rad)
                        R_trial = current_rotations[gid] @ R_delta
                        # Translation delta scaled by amplitude (so
                        # fine-amplitude phases also fine-tune the
                        # lattice anchor).
                        frac = rng.uniform(-1.0, 1.0, size=3)
                        T_delta = trans_scale * (frac @ translation_basis)
                        T_trial = current_translations[gid] + T_delta

                        retile = _retile_grain(
                            master_positions=master_pos,
                            master_numbers=master_num,
                            voronoi_cell=voronoi_cell,
                            seed_world=seed_world,
                            box_dim=box_dim,
                            rotation=R_trial,
                            translation=T_trial,
                            target_n=target_n,
                        )
                        if retile is None:
                            self.atoms.positions = snapshot.copy()
                            continue
                        new_pos_world, _ = retile

                        self.atoms.positions = snapshot.copy()
                        self.atoms.positions[grain_mask] = new_pos_world

                        self.shell_relax(
                            shell_target,
                            num_steps=int(local_fire_steps),
                            freeze_mask=freeze_mask,
                            neighbor_update_interval=99999,
                            capture_trajectory=False,
                            show_progress=False,
                            **weights,
                        )

                        cost = _global_cost(
                            self, shell_target, weights,
                        )["total"]
                        if cost < best_cost:
                            best_cost = cost
                            best_R = R_trial
                            best_T = T_trial
                            best_atoms = self.atoms.positions.copy()

                        self.atoms.positions = snapshot.copy()

                    self.atoms.positions = snapshot.copy()
                    if best_atoms is not None:
                        self.atoms.positions = best_atoms
                        current_rotations[gid] = best_R
                        current_translations[gid] = best_T
                        iteration += 1
                        improved_this_round = True
                        _record(iteration, int(gid),
                                float(amplitude_deg), phase_idx)
                        if show_progress:
                            elapsed = time.time() - t0
                            print(
                                f"  amp {amplitude_deg:5.1f}° "
                                f"round {round_idx+1} grain {gid}: "
                                f"cost {pre_cost:.4f} -> {best_cost:.4f} "
                                f"(elapsed {elapsed:.1f} s)"
                            )

                if not improved_this_round:
                    # No grain found an improvement at this amplitude
                    # round → skip remaining rounds, drop to next
                    # (smaller) amplitude.
                    if show_progress:
                        print(
                            f"  amp {amplitude_deg:5.1f}° "
                            f"round {round_idx+1}: no improvements"
                        )
                    break

        # Final whole-cell FIRE quench
        if final_quench_steps > 0:
            if show_progress:
                print(f"  final FIRE quench ({final_quench_steps} steps)...")
            self.shell_relax(
                shell_target,
                num_steps=int(final_quench_steps),
                k_restraint=0.0,
                capture_trajectory=False,
                show_progress=False,
                **weights,
            )
            iteration += 1
            _record(iteration, -2, 0.0, -1)

        traj = (
            np.asarray(history["trajectory"], dtype=np.float32)
            if capture_trajectory else None
        )
        history_out = dict(
            iteration=np.asarray(history["iteration"], dtype=np.intp),
            global_cost=np.asarray(history["global_cost"], dtype=np.float64),
            cost_bond=np.asarray(history["cost_bond"], dtype=np.float64),
            cost_angle=np.asarray(history["cost_angle"], dtype=np.float64),
            cost_rep=np.asarray(history["cost_rep"], dtype=np.float64),
            accepted_grain=np.asarray(history["accepted_grain"], dtype=np.intp),
            rotation_amplitude_deg=np.asarray(
                history["rotation_amplitude_deg"], dtype=np.float64,
            ),
            amplitude_phase=np.asarray(
                history["amplitude_phase"], dtype=np.intp,
            ),
            # Compat with old `pass_index` consumers (cost plot uses
            # this for vertical pass-boundary lines and for spotting
            # the final FIRE quench at -1).
            pass_index=np.asarray(
                history["amplitude_phase"], dtype=np.intp,
            ),
            trajectory=traj,
        )
        self.refine_grains_history = history_out
        self.current_distribution = None
        if hasattr(self, "_rebuild_spatial_index"):
            self._rebuild_spatial_index()
        return history_out

    # =====================================================================
    # refine_grains_v2 — fresh-baseline architecture
    # =====================================================================
    #
    # The earlier ``refine_grains`` / ``refine_grains_coarse_to_fine``
    # methods compared trials against the *post-generate* (whole-cell-
    # FIRE-relaxed) cost.  Diagnostics on NC 40³ showed this is
    # fundamentally unfair: re-tiling a grain (even with the identity
    # rotation) places atoms back at *unrelaxed* lattice positions, so
    # trials always start ~7 Å away from the relaxed state and
    # constrained FIRE can only chip a small amount off.  Result: 0
    # accepts on 40³.
    #
    # The fix is to make the *baseline* go through the same retile +
    # FIRE process as the trials.  We discard generate's whole-cell
    # FIRE relaxation, retile every grain at its initial rotation, run
    # one per-grain FIRE pass to warm up each grain's neighbourhood,
    # and compare future trials against this freshly-converged state.
    # Now apples-to-apples by construction.
    #
    # Phases:
    #   A. Reset to as-built.   Retile each grain with R_initial, T=0.
    #   B. Per-grain warmup.    For each grain, FIRE on grain + ~1.5
    #                           pair-peak neighbours.  Sets the
    #                           apples-to-apples ``baseline_cost``.
    #   C. Coarse refinement.   Random SO(3) rotations + translations
    #                           per grain, retile + FIRE on
    #                           neighbourhood, accept if global cost
    #                           drops.
    #   D. Fine refinement.     Same but with bounded amplitude (~5°)
    #                           composed onto the grain's current
    #                           orientation.
    #   E. Whole-cell quench.   Final FIRE on the full cell.
    def refine_grains_v2(
        self: "Supercell",
        shell_target: "CoordinationShellTarget",
        *,
        # Phase B: per-grain warmup (only applies when
        # ``whole_cell_trials=False`` — whole-cell mode skips A/B and
        # uses the post-generate state directly as the baseline).
        warmup_fire_steps: int = 50,
        neighbor_radius_factor: float = 2.0,
        # Phase C: coarse (random SO(3)) refinement
        coarse_trials_per_grain: int = 16,
        coarse_max_passes: int = 4,
        coarse_fire_steps: int = 50,
        # Phase D: fine (bounded amplitude) refinement.  Walks the
        # ``fine_amplitude_schedule_deg`` from coarse to fine; at each
        # amplitude does up to ``fine_max_passes`` round-robin passes
        # over grains, breaking when a pass finds no accepts.  When
        # ``fine_amplitude_schedule_deg`` is None, falls back to a
        # single amplitude given by ``fine_amplitude_deg``.
        fine_amplitude_deg: float = 5.0,
        fine_amplitude_schedule_deg: "tuple[float, ...] | None" = None,
        fine_trials_per_grain: int = 8,
        fine_max_passes: int = 4,
        fine_fire_steps: int = 50,
        # Phase E: whole-cell quench
        final_quench_steps: int = 200,
        # Spring weights (passed through to shell_relax & cost)
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 3.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        # Trial FIRE strategy.
        #   False (default): per-grain warmup baseline + constrained
        #     trials.  Cheap but selects orientations by a
        #     per-grain-FIRE criterion that doesn't track whole-cell
        #     quality on big cells (NC 40³ converges to a worse final
        #     state than ``generate()`` alone).
        #   True: skip phase A/B; baseline = post-generate cost; every
        #     trial runs whole-cell FIRE just like ``generate()`` did.
        #     Apples-to-apples with the final state, but ~3-4× slower
        #     per trial.
        whole_cell_trials: bool = False,
        # Output
        time_budget_sec: float = 300.0,
        capture_trajectory: bool = True,
        show_progress: bool = True,
        rng_seed: "int | None" = None,
    ) -> dict:
        """Fresh-baseline grain refinement (the v2 architecture).

        See the long block comment above this method for the design.

        Returns
        -------
        history : dict
            Same shape as ``refine_grains_coarse_to_fine``'s history,
            so the existing trajectory + cost-plot helpers work
            unchanged.  Phase index in ``pass_index`` /
            ``amplitude_phase``: ``0`` = warmup baseline, ``1`` =
            coarse-pass index, ``-2`` (encoded) = fine-pass index, ``-1``
            = final quench.
        """
        if getattr(self, "_grain_ids", None) is None:
            raise ValueError(
                "refine_grains_v2 requires a grain-built cell.  Call "
                "Supercell.generate(grain_size=...) first.")
        if getattr(self, "_grain_cells", None) is None:
            raise ValueError(
                "refine_grains_v2 requires Voronoi cells cached on the "
                "supercell.  Re-run generate() to populate them.")

        rng = (np.random.default_rng(rng_seed)
               if rng_seed is not None else self.rng)

        weights = dict(
            bond_weight=float(bond_weight),
            angle_weight=float(angle_weight),
            repulsion_weight=float(repulsion_weight),
            hard_core_scale=float(hard_core_scale),
            nonbond_push_scale=float(nonbond_push_scale),
        )

        grain_ids = np.asarray(self._grain_ids, dtype=np.intp)
        grain_seeds = np.asarray(self._grain_seeds, dtype=np.float64)
        voronoi_cells = self._grain_cells
        masters = self._grain_masters
        grain_source = self._grain_source
        if grain_source is None:
            grain_source = np.zeros(len(grain_seeds), dtype=np.intp)
        is_crystalline = np.asarray(
            self._grain_is_crystalline, dtype=bool,
        )
        box_dim = np.asarray(self._grain_box_dim, dtype=np.float64)
        master_lattice = np.asarray(
            self._grain_master_lattice, dtype=np.float64,
        )
        translation_basis = master_lattice * 0.5

        pair_peak_max = float(np.max(
            np.asarray(shell_target.pair_peak, dtype=np.float64)
        ))
        neighborhood_radius = float(
            neighbor_radius_factor * pair_peak_max
        )

        cell_mat = np.ascontiguousarray(
            self.atoms.cell.array, dtype=np.float64,
        )
        cell_inv = np.linalg.inv(cell_mat)

        unique_grains = [
            int(g) for g in np.unique(grain_ids[grain_ids >= 0])
            if is_crystalline[int(g)]
        ]
        if not unique_grains:
            raise ValueError(
                "No crystalline grains found.  refine_grains_v2 has "
                "nothing to refine.")

        max_grain_id = int(grain_ids.max()) + 1
        current_rotations = np.zeros((max_grain_id, 3, 3), dtype=np.float64)
        for g in range(max_grain_id):
            current_rotations[g] = (
                self._grain_rotations_initial[g]
                if g < len(self._grain_rotations_initial)
                else np.eye(3)
            )
        current_translations = np.zeros((max_grain_id, 3), dtype=np.float64)

        # Per-grain master/voronoi lookup helper (reused below).
        def _grain_meta(gid: int):
            grain_mask = (grain_ids == gid)
            target_n = int(np.sum(grain_mask))
            seed_world = grain_seeds[gid]
            voronoi_cell = voronoi_cells[gid]
            src_idx = int(grain_source[gid])
            master = masters[src_idx]
            master_pos = np.asarray(
                master["positions"], dtype=np.float64,
            )
            master_num = np.asarray(
                master["numbers"], dtype=np.int64,
            )
            return (grain_mask, target_n, seed_world, voronoi_cell,
                    master_pos, master_num)

        history: dict = dict(
            iteration=[], global_cost=[],
            cost_bond=[], cost_angle=[], cost_rep=[],
            accepted_grain=[],
            rotation_amplitude_deg=[],
            amplitude_phase=[],
            trajectory=[] if capture_trajectory else None,
        )

        def _record(it_idx, accepted_g, amp_deg, phase_idx):
            cost = _global_cost(self, shell_target, weights)
            history["iteration"].append(it_idx)
            history["global_cost"].append(cost["total"])
            history["cost_bond"].append(cost["bond"])
            history["cost_angle"].append(cost["angle"])
            history["cost_rep"].append(cost["rep"])
            history["accepted_grain"].append(accepted_g)
            history["rotation_amplitude_deg"].append(amp_deg)
            history["amplitude_phase"].append(phase_idx)
            if capture_trajectory:
                history["trajectory"].append(
                    self.atoms.positions.copy().astype(np.float32)
                )

        if not whole_cell_trials:
            # =====================================================
            # PHASE A — Reset every grain to its as-built (lattice)
            # state.  This discards generate's whole-cell FIRE so
            # the per-grain warmup that follows produces an
            # apples-to-apples baseline with constrained trials.
            # =====================================================
            if show_progress:
                print(
                    "refine_grains_v2: phase A — reset to as-built "
                    "(retile every grain at R_initial, T=0)"
                )
            for gid in unique_grains:
                (grain_mask, target_n, seed_world, voronoi_cell,
                 master_pos, master_num) = _grain_meta(gid)
                retile = _retile_grain(
                    master_positions=master_pos, master_numbers=master_num,
                    voronoi_cell=voronoi_cell, seed_world=seed_world,
                    box_dim=box_dim,
                    rotation=current_rotations[gid],
                    translation=current_translations[gid],
                    target_n=target_n,
                )
                if retile is None:
                    raise RuntimeError(
                        f"phase A: retile failed for grain {gid} at "
                        "its initial rotation — should never happen."
                    )
                new_pos_world, _ = retile
                self.atoms.positions[grain_mask] = new_pos_world

            if hasattr(self, "_rebuild_spatial_index"):
                self._rebuild_spatial_index()

            # =====================================================
            # PHASE B — Per-grain warmup FIRE.  For each grain,
            # release its neighbourhood and run X FIRE steps.  After
            # this loop the cell is at the per-grain-FIRE-converged
            # surface that trials will compete against.
            # =====================================================
            if show_progress:
                print(
                    f"refine_grains_v2: phase B — per-grain warmup "
                    f"({warmup_fire_steps} FIRE on grain + "
                    f"{neighbor_radius_factor}·pair_peak neighbours)"
                )
            for gid in unique_grains:
                grain_mask = (grain_ids == gid)
                neighborhood_mask = _expand_neighborhood(
                    grain_mask=grain_mask,
                    positions=self.atoms.positions,
                    cell_mat=cell_mat, cell_inv=cell_inv,
                    radius=neighborhood_radius,
                )
                freeze_mask = ~neighborhood_mask
                self.shell_relax(
                    shell_target,
                    num_steps=int(warmup_fire_steps),
                    freeze_mask=freeze_mask,
                    neighbor_update_interval=99999,
                    capture_trajectory=False,
                    show_progress=False,
                    **weights,
                )
        else:
            if show_progress:
                print(
                    "refine_grains_v2: whole-cell mode — skipping "
                    "phase A/B; baseline = post-generate cost"
                )

        baseline_cost = _global_cost(
            self, shell_target, weights,
        )["total"]
        if show_progress:
            print(f"  baseline: {baseline_cost:.4f}")
        _record(0, -1, 0.0, 0)

        iteration = 0
        t0 = time.time()

        # =====================================================
        # PHASE C — Coarse refinement, random SO(3).
        # =====================================================
        if show_progress:
            print(
                "refine_grains_v2: phase C — coarse refinement "
                f"({coarse_max_passes} passes × "
                f"{coarse_trials_per_grain} trials/grain, random "
                "SO(3))"
            )
        for pass_idx in range(coarse_max_passes):
            if time.time() - t0 >= time_budget_sec:
                break
            grain_order = list(unique_grains)
            rng.shuffle(grain_order)
            improved_this_pass = False

            for gid in grain_order:
                if time.time() - t0 >= time_budget_sec:
                    break
                (grain_mask, target_n, seed_world, voronoi_cell,
                 master_pos, master_num) = _grain_meta(gid)
                snapshot = self.atoms.positions.copy()
                pre_cost = baseline_cost
                best_cost = pre_cost
                best_R = None
                best_T = None
                best_atoms: "np.ndarray | None" = None

                neighborhood_mask = _expand_neighborhood(
                    grain_mask=grain_mask,
                    positions=self.atoms.positions,
                    cell_mat=cell_mat, cell_inv=cell_inv,
                    radius=neighborhood_radius,
                )
                freeze_mask = ~neighborhood_mask

                for _trial in range(coarse_trials_per_grain):
                    R_trial = _so3_random_rotation(
                        rng,
                        angle_min_rad=np.deg2rad(10.0),
                        angle_max_rad=np.pi,
                    )
                    frac = rng.uniform(-1.0, 1.0, size=3)
                    T_trial = frac @ translation_basis
                    retile = _retile_grain(
                        master_positions=master_pos,
                        master_numbers=master_num,
                        voronoi_cell=voronoi_cell,
                        seed_world=seed_world, box_dim=box_dim,
                        rotation=R_trial, translation=T_trial,
                        target_n=target_n,
                    )
                    if retile is None:
                        self.atoms.positions = snapshot.copy()
                        continue
                    new_pos_world, _ = retile
                    self.atoms.positions = snapshot.copy()
                    self.atoms.positions[grain_mask] = new_pos_world
                    self.shell_relax(
                        shell_target,
                        num_steps=int(coarse_fire_steps),
                        freeze_mask=(None if whole_cell_trials
                                     else freeze_mask),
                        neighbor_update_interval=99999,
                        capture_trajectory=False,
                        show_progress=False,
                        **weights,
                    )
                    cost = _global_cost(
                        self, shell_target, weights,
                    )["total"]
                    if cost < best_cost:
                        best_cost = cost
                        best_R = R_trial
                        best_T = T_trial
                        best_atoms = self.atoms.positions.copy()
                    self.atoms.positions = snapshot.copy()

                self.atoms.positions = snapshot.copy()
                if best_atoms is not None:
                    self.atoms.positions = best_atoms
                    current_rotations[gid] = best_R
                    current_translations[gid] = best_T
                    baseline_cost = best_cost
                    iteration += 1
                    improved_this_pass = True
                    _record(iteration, int(gid), 180.0, pass_idx + 1)
                    if show_progress:
                        elapsed = time.time() - t0
                        print(
                            f"  C pass {pass_idx+1} grain {gid}: "
                            f"{pre_cost:.4f} -> {best_cost:.4f} "
                            f"(elapsed {elapsed:.1f} s)"
                        )

            if not improved_this_pass:
                if show_progress:
                    print(
                        f"  C pass {pass_idx+1}: no improvements, "
                        "moving to phase D"
                    )
                break

        # =====================================================
        # PHASE D — Fine refinement, bounded amplitude (schedule).
        # =====================================================
        amp_schedule = (
            tuple(fine_amplitude_schedule_deg)
            if fine_amplitude_schedule_deg is not None
            else (float(fine_amplitude_deg),)
        )
        if show_progress:
            print(
                "refine_grains_v2: phase D — fine refinement "
                f"(schedule={amp_schedule}, {fine_max_passes} "
                f"passes × {fine_trials_per_grain} trials/grain)"
            )
        for amp_step, amp_deg in enumerate(amp_schedule):
            if time.time() - t0 >= time_budget_sec:
                break
            fine_amp_rad = np.deg2rad(float(amp_deg))
            fine_trans_scale = float(amp_deg / 180.0)
            broke_for_no_improvement = False
            for pass_idx in range(fine_max_passes):
                if time.time() - t0 >= time_budget_sec:
                    break
                grain_order = list(unique_grains)
                rng.shuffle(grain_order)
                improved_this_pass = False

                for gid in grain_order:
                    if time.time() - t0 >= time_budget_sec:
                        break
                    (grain_mask, target_n, seed_world, voronoi_cell,
                     master_pos, master_num) = _grain_meta(gid)
                    snapshot = self.atoms.positions.copy()
                    pre_cost = baseline_cost
                    best_cost = pre_cost
                    best_R = None
                    best_T = None
                    best_atoms = None

                    neighborhood_mask = _expand_neighborhood(
                        grain_mask=grain_mask,
                        positions=self.atoms.positions,
                        cell_mat=cell_mat, cell_inv=cell_inv,
                        radius=neighborhood_radius,
                    )
                    freeze_mask = ~neighborhood_mask

                    for _trial in range(fine_trials_per_grain):
                        R_delta = _so3_bounded_rotation(rng, fine_amp_rad)
                        R_trial = current_rotations[gid] @ R_delta
                        frac = rng.uniform(-1.0, 1.0, size=3)
                        T_delta = fine_trans_scale * (frac @ translation_basis)
                        T_trial = current_translations[gid] + T_delta
                        retile = _retile_grain(
                            master_positions=master_pos,
                            master_numbers=master_num,
                            voronoi_cell=voronoi_cell,
                            seed_world=seed_world, box_dim=box_dim,
                            rotation=R_trial, translation=T_trial,
                            target_n=target_n,
                        )
                        if retile is None:
                            self.atoms.positions = snapshot.copy()
                            continue
                        new_pos_world, _ = retile
                        self.atoms.positions = snapshot.copy()
                        self.atoms.positions[grain_mask] = new_pos_world
                        self.shell_relax(
                            shell_target,
                            num_steps=int(fine_fire_steps),
                            freeze_mask=(None if whole_cell_trials
                                         else freeze_mask),
                            neighbor_update_interval=99999,
                            capture_trajectory=False,
                            show_progress=False,
                            **weights,
                        )
                        cost = _global_cost(
                            self, shell_target, weights,
                        )["total"]
                        if cost < best_cost:
                            best_cost = cost
                            best_R = R_trial
                            best_T = T_trial
                            best_atoms = self.atoms.positions.copy()
                        self.atoms.positions = snapshot.copy()

                    self.atoms.positions = snapshot.copy()
                    if best_atoms is not None:
                        self.atoms.positions = best_atoms
                        current_rotations[gid] = best_R
                        current_translations[gid] = best_T
                        baseline_cost = best_cost
                        iteration += 1
                        improved_this_pass = True
                        # Encode fine-phase index by amp_step + pass_idx
                        # offset (so the cost plotter renders amplitude
                        # phase boundaries correctly).
                        fine_phase = (
                            coarse_max_passes
                            + 1
                            + amp_step * fine_max_passes
                            + pass_idx
                        )
                        _record(iteration, int(gid),
                                float(amp_deg), fine_phase)
                        if show_progress:
                            elapsed = time.time() - t0
                            print(
                                f"  D amp {amp_deg:.1f}° pass "
                                f"{pass_idx+1} grain {gid}: "
                                f"{pre_cost:.4f} -> {best_cost:.4f} "
                                f"(elapsed {elapsed:.1f} s)"
                            )

                if not improved_this_pass:
                    if show_progress:
                        print(
                            f"  D amp {amp_deg:.1f}° pass "
                            f"{pass_idx+1}: no improvements, "
                            "next amplitude"
                        )
                    broke_for_no_improvement = True
                    break

            if broke_for_no_improvement:
                # No improvements at this amplitude -> keep walking
                # the schedule (smaller amplitudes might still find
                # subtle accepts).
                continue

        # =====================================================
        # PHASE E — Whole-cell FIRE quench.
        # =====================================================
        if final_quench_steps > 0:
            if show_progress:
                print(
                    f"refine_grains_v2: phase E — whole-cell FIRE "
                    f"quench ({final_quench_steps} steps)"
                )
            self.shell_relax(
                shell_target,
                num_steps=int(final_quench_steps),
                k_restraint=0.0,
                capture_trajectory=False,
                show_progress=False,
                **weights,
            )
            iteration += 1
            _record(iteration, -2, 0.0, -1)

        traj = (
            np.asarray(history["trajectory"], dtype=np.float32)
            if capture_trajectory else None
        )
        history_out = dict(
            iteration=np.asarray(history["iteration"], dtype=np.intp),
            global_cost=np.asarray(history["global_cost"], dtype=np.float64),
            cost_bond=np.asarray(history["cost_bond"], dtype=np.float64),
            cost_angle=np.asarray(history["cost_angle"], dtype=np.float64),
            cost_rep=np.asarray(history["cost_rep"], dtype=np.float64),
            accepted_grain=np.asarray(history["accepted_grain"], dtype=np.intp),
            rotation_amplitude_deg=np.asarray(
                history["rotation_amplitude_deg"], dtype=np.float64,
            ),
            amplitude_phase=np.asarray(
                history["amplitude_phase"], dtype=np.intp,
            ),
            pass_index=np.asarray(
                history["amplitude_phase"], dtype=np.intp,
            ),
            trajectory=traj,
        )
        self.refine_grains_history = history_out
        self.current_distribution = None
        if hasattr(self, "_rebuild_spatial_index"):
            self._rebuild_spatial_index()
        return history_out

    # =====================================================================
    # refine_grains_v3 — user's "per-grain warmup baseline" plan, tight
    # =====================================================================
    #
    # Implements the 7-step algorithm sketched by the user, but with
    # tight defaults aimed at finding many accepts fast on small cells:
    #
    #   1-2: spherical master + sphere-rotate-shift-crop per grain.
    #        Already done by ``Supercell.generate(grain_size=...)``;
    #        ``_grain_masters`` is a sphere of seed-local radius equal
    #        to the largest seed-to-vertex distance, so any rotation
    #        keeps the cell fully covered.
    #   3:   "store as initial config" = the post-generate state.  We
    #        do *not* reset positions — the post-generate state is
    #        already at the constrained-FIRE minimum for low grain
    #        counts (verified on NC 20³), so a reset just throws away
    #        the relaxation we'd be re-discovering anyway.
    #   4:   per-grain warmup FIRE on grain + ``neighbor_radius_factor
    #        × pair_peak`` neighbours, used to set ``baseline_cost``.
    #        On small cells this is a near-no-op (already at the
    #        constrained min) but it gives the trial loop a fair
    #        starting point on bigger cells.
    #   5-6: a phase-5 random-SO(3) sweep then a phase-6 bounded-
    #        amplitude schedule (45° → 22° → 11° → 5° → 2° → 1° by
    #        default).  Each trial: pick a grain, retile with the
    #        spherical master at the trial rotation, run
    #        ``fire_steps_per_trial`` constrained FIRE on its
    #        neighbourhood, accept iff the *global* cost drops below
    #        ``baseline_cost``.
    #   7:   final whole-cell FIRE quench.
    def refine_grains_v3(
        self: "Supercell",
        shell_target: "CoordinationShellTarget",
        *,
        # Phase 4: per-grain warmup
        warmup_fire_steps: int = 30,
        neighbor_radius_factor: float = 1.5,
        # Trial FIRE strategy: ``True`` means trials run whole-cell
        # FIRE (freeze_mask=None) — required for small cells where the
        # constrained-FIRE basin lands the trial above the
        # whole-cell-FIRE baseline.
        whole_cell_trials: bool = True,
        # Phase 5/6 — random SO(3) (coarse) + amplitude schedule (fine)
        coarse_trials_per_grain: int = 8,
        coarse_max_passes: int = 1,
        fine_amplitude_schedule_deg: tuple = (45.0, 22.0, 11.0, 5.0,
                                              2.0, 1.0),
        fine_trials_per_grain: int = 8,
        fine_max_passes: int = 3,
        fire_steps_per_trial: int = 30,
        # Phase 7
        final_quench_steps: int = 100,
        # Spring weights (passed to shell_relax + cost)
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 3.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        # Output
        time_budget_sec: float = 60.0,
        capture_trajectory: bool = True,
        show_progress: bool = True,
        rng_seed: "int | None" = None,
    ) -> dict:
        """Tight implementation of the user's 7-step algorithm.

        Returns a history dict in the same format as
        ``refine_grains_coarse_to_fine``.
        """
        if getattr(self, "_grain_ids", None) is None:
            raise ValueError(
                "refine_grains_v3 requires a grain-built cell.  "
                "Call Supercell.generate(grain_size=...) first.")
        if getattr(self, "_grain_cells", None) is None:
            raise ValueError(
                "refine_grains_v3 requires Voronoi cells cached on "
                "the supercell.  Re-run generate() to populate them.")

        rng = (np.random.default_rng(rng_seed)
               if rng_seed is not None else self.rng)

        weights = dict(
            bond_weight=float(bond_weight),
            angle_weight=float(angle_weight),
            repulsion_weight=float(repulsion_weight),
            hard_core_scale=float(hard_core_scale),
            nonbond_push_scale=float(nonbond_push_scale),
        )

        grain_ids = np.asarray(self._grain_ids, dtype=np.intp)
        grain_seeds = np.asarray(self._grain_seeds, dtype=np.float64)
        voronoi_cells = self._grain_cells
        masters = self._grain_masters
        grain_source = self._grain_source
        if grain_source is None:
            grain_source = np.zeros(len(grain_seeds), dtype=np.intp)
        is_crystalline = np.asarray(
            self._grain_is_crystalline, dtype=bool,
        )
        box_dim = np.asarray(self._grain_box_dim, dtype=np.float64)
        master_lattice = np.asarray(
            self._grain_master_lattice, dtype=np.float64,
        )
        translation_basis = master_lattice * 0.5

        pair_peak_max = float(np.max(
            np.asarray(shell_target.pair_peak, dtype=np.float64)
        ))
        neighborhood_radius = float(
            neighbor_radius_factor * pair_peak_max
        )

        cell_mat = np.ascontiguousarray(
            self.atoms.cell.array, dtype=np.float64,
        )
        cell_inv = np.linalg.inv(cell_mat)

        unique_grains = [
            int(g) for g in np.unique(grain_ids[grain_ids >= 0])
            if is_crystalline[int(g)]
        ]
        if not unique_grains:
            raise ValueError(
                "No crystalline grains found.  refine_grains_v3 has "
                "nothing to refine.")

        max_grain_id = int(grain_ids.max()) + 1
        current_rotations = np.zeros((max_grain_id, 3, 3),
                                     dtype=np.float64)
        for g in range(max_grain_id):
            current_rotations[g] = (
                self._grain_rotations_initial[g]
                if g < len(self._grain_rotations_initial)
                else np.eye(3)
            )
        current_translations = np.zeros((max_grain_id, 3),
                                        dtype=np.float64)

        def _grain_meta(gid: int):
            grain_mask = (grain_ids == gid)
            target_n = int(np.sum(grain_mask))
            seed_world = grain_seeds[gid]
            voronoi_cell = voronoi_cells[gid]
            src_idx = int(grain_source[gid])
            master = masters[src_idx]
            master_pos = np.asarray(
                master["positions"], dtype=np.float64,
            )
            master_num = np.asarray(
                master["numbers"], dtype=np.int64,
            )
            return (grain_mask, target_n, seed_world, voronoi_cell,
                    master_pos, master_num)

        history: dict = dict(
            iteration=[], global_cost=[],
            cost_bond=[], cost_angle=[], cost_rep=[],
            accepted_grain=[],
            rotation_amplitude_deg=[],
            amplitude_phase=[],
            trajectory=[] if capture_trajectory else None,
        )

        def _record(it_idx, accepted_g, amp_deg, phase_idx):
            cost = _global_cost(self, shell_target, weights)
            history["iteration"].append(it_idx)
            history["global_cost"].append(cost["total"])
            history["cost_bond"].append(cost["bond"])
            history["cost_angle"].append(cost["angle"])
            history["cost_rep"].append(cost["rep"])
            history["accepted_grain"].append(accepted_g)
            history["rotation_amplitude_deg"].append(amp_deg)
            history["amplitude_phase"].append(phase_idx)
            if capture_trajectory:
                history["trajectory"].append(
                    self.atoms.positions.copy().astype(np.float32)
                )

        # ─── Phase 4 — per-grain warmup, apples-to-apples with the
        #              trial protocol.  When trials run whole-cell
        #              FIRE the warmup ALSO runs whole-cell FIRE (and
        #              we just do one short pass, since generate
        #              already converged the post-generate state).
        # ────────────────────────────────────────────────────────
        if show_progress:
            mode = "whole-cell" if whole_cell_trials else "constrained"
            print(
                f"refine_grains_v3: phase 4 — warmup ({mode} FIRE, "
                f"{warmup_fire_steps} steps)"
            )
        if whole_cell_trials:
            if warmup_fire_steps > 0:
                self.shell_relax(
                    shell_target,
                    num_steps=int(warmup_fire_steps),
                    freeze_mask=None,
                    neighbor_update_interval=99999,
                    capture_trajectory=False,
                    show_progress=False,
                    **weights,
                )
        else:
            for gid in unique_grains:
                grain_mask = (grain_ids == gid)
                neighborhood_mask = _expand_neighborhood(
                    grain_mask=grain_mask,
                    positions=self.atoms.positions,
                    cell_mat=cell_mat, cell_inv=cell_inv,
                    radius=neighborhood_radius,
                )
                self.shell_relax(
                    shell_target,
                    num_steps=int(warmup_fire_steps),
                    freeze_mask=~neighborhood_mask,
                    neighbor_update_interval=99999,
                    capture_trajectory=False,
                    show_progress=False,
                    **weights,
                )
        baseline_cost = _global_cost(
            self, shell_target, weights,
        )["total"]
        if show_progress:
            print(f"  baseline (post-warmup): {baseline_cost:.4f}")
        _record(0, -1, 0.0, 0)

        iteration = 0
        t0 = time.time()

        # Helper: trial loop for one grain at a given (rotation
        # sampler, amplitude_label, phase_index).
        def _trial_pass(
            gid: int, rotation_sampler, amplitude_label: float,
            phase_idx: int, n_trials: int,
        ):
            nonlocal iteration, baseline_cost
            (grain_mask, target_n, seed_world, voronoi_cell,
             master_pos, master_num) = _grain_meta(gid)
            snapshot = self.atoms.positions.copy()
            best_cost = baseline_cost
            best_R = None
            best_T = None
            best_atoms: "np.ndarray | None" = None

            neighborhood_mask = _expand_neighborhood(
                grain_mask=grain_mask,
                positions=self.atoms.positions,
                cell_mat=cell_mat, cell_inv=cell_inv,
                radius=neighborhood_radius,
            )
            freeze_mask = ~neighborhood_mask

            for _ in range(n_trials):
                R_trial, T_trial = rotation_sampler(
                    current_rotations[gid], current_translations[gid],
                )
                retile = _retile_grain(
                    master_positions=master_pos,
                    master_numbers=master_num,
                    voronoi_cell=voronoi_cell,
                    seed_world=seed_world, box_dim=box_dim,
                    rotation=R_trial, translation=T_trial,
                    target_n=target_n,
                )
                if retile is None:
                    self.atoms.positions = snapshot.copy()
                    continue
                new_pos_world, _ = retile
                self.atoms.positions = snapshot.copy()
                self.atoms.positions[grain_mask] = new_pos_world
                self.shell_relax(
                    shell_target,
                    num_steps=int(fire_steps_per_trial),
                    freeze_mask=(None if whole_cell_trials
                                 else freeze_mask),
                    neighbor_update_interval=99999,
                    capture_trajectory=False,
                    show_progress=False,
                    **weights,
                )
                cost = _global_cost(
                    self, shell_target, weights,
                )["total"]
                if cost < best_cost:
                    best_cost = cost
                    best_R = R_trial
                    best_T = T_trial
                    best_atoms = self.atoms.positions.copy()
                self.atoms.positions = snapshot.copy()

            self.atoms.positions = snapshot.copy()
            if best_atoms is not None:
                self.atoms.positions = best_atoms
                current_rotations[gid] = best_R
                current_translations[gid] = best_T
                baseline_cost = best_cost
                iteration += 1
                _record(iteration, int(gid),
                        float(amplitude_label), phase_idx)
                return True
            return False

        # ─── Phase 5 — random SO(3) ─────────────────────────────────
        if show_progress:
            print(
                f"refine_grains_v3: phase 5 — random SO(3) "
                f"({coarse_max_passes} passes × "
                f"{coarse_trials_per_grain} trials/grain)"
            )

        def _uniform_sampler(R_curr, T_curr):
            R = _so3_random_rotation(
                rng,
                angle_min_rad=np.deg2rad(10.0),
                angle_max_rad=np.pi,
            )
            frac = rng.uniform(-1.0, 1.0, size=3)
            return R, frac @ translation_basis

        for pass_idx in range(coarse_max_passes):
            if time.time() - t0 >= time_budget_sec:
                break
            grain_order = list(unique_grains)
            rng.shuffle(grain_order)
            improved = False
            for gid in grain_order:
                if time.time() - t0 >= time_budget_sec:
                    break
                if _trial_pass(
                    gid, _uniform_sampler, 180.0, pass_idx + 1,
                    coarse_trials_per_grain,
                ):
                    improved = True
                    if show_progress:
                        elapsed = time.time() - t0
                        print(
                            f"  5 pass {pass_idx+1} grain {gid}: "
                            f"cost -> {baseline_cost:.4f} "
                            f"(elapsed {elapsed:.1f} s)"
                        )
            if not improved:
                if show_progress:
                    print(
                        f"  5 pass {pass_idx+1}: no improvements, "
                        "moving to phase 6"
                    )
                break

        # ─── Phase 6 — bounded amplitude schedule ──────────────────
        if show_progress:
            print(
                f"refine_grains_v3: phase 6 — amplitude schedule "
                f"{tuple(fine_amplitude_schedule_deg)} "
                f"({fine_max_passes} passes × "
                f"{fine_trials_per_grain} trials/grain)"
            )
        for amp_step, amp_deg in enumerate(fine_amplitude_schedule_deg):
            if time.time() - t0 >= time_budget_sec:
                break
            amp_rad = np.deg2rad(float(amp_deg))
            trans_scale = float(amp_deg / 180.0)

            def _bounded_sampler(R_curr, T_curr,
                                 _amp=amp_rad, _ts=trans_scale):
                R_delta = _so3_bounded_rotation(rng, _amp)
                R = R_curr @ R_delta
                frac = rng.uniform(-1.0, 1.0, size=3)
                T = T_curr + _ts * (frac @ translation_basis)
                return R, T

            for pass_idx in range(fine_max_passes):
                if time.time() - t0 >= time_budget_sec:
                    break
                grain_order = list(unique_grains)
                rng.shuffle(grain_order)
                improved = False
                phase_idx = (
                    coarse_max_passes + 1
                    + amp_step * fine_max_passes + pass_idx
                )
                for gid in grain_order:
                    if time.time() - t0 >= time_budget_sec:
                        break
                    if _trial_pass(
                        gid, _bounded_sampler, float(amp_deg),
                        phase_idx, fine_trials_per_grain,
                    ):
                        improved = True
                        if show_progress:
                            elapsed = time.time() - t0
                            print(
                                f"  6 amp {amp_deg:.1f}° pass "
                                f"{pass_idx+1} grain {gid}: cost -> "
                                f"{baseline_cost:.4f} "
                                f"(elapsed {elapsed:.1f} s)"
                            )
                if not improved and show_progress:
                    print(
                        f"  6 amp {amp_deg:.1f}° pass "
                        f"{pass_idx+1}: no improvements"
                    )
                # Don't break on no-improvement — let the time budget
                # control the loop.  Re-running the same amplitude
                # with a fresh RNG state often finds new accepts.

        # ─── Phase 7 — final whole-cell FIRE quench ────────────────
        if final_quench_steps > 0:
            if show_progress:
                print(
                    f"refine_grains_v3: phase 7 — whole-cell FIRE "
                    f"quench ({final_quench_steps} steps)"
                )
            self.shell_relax(
                shell_target,
                num_steps=int(final_quench_steps),
                k_restraint=0.0,
                capture_trajectory=False,
                show_progress=False,
                **weights,
            )
            iteration += 1
            _record(iteration, -2, 0.0, -1)

        traj = (
            np.asarray(history["trajectory"], dtype=np.float32)
            if capture_trajectory else None
        )
        history_out = dict(
            iteration=np.asarray(history["iteration"], dtype=np.intp),
            global_cost=np.asarray(history["global_cost"],
                                   dtype=np.float64),
            cost_bond=np.asarray(history["cost_bond"], dtype=np.float64),
            cost_angle=np.asarray(history["cost_angle"], dtype=np.float64),
            cost_rep=np.asarray(history["cost_rep"], dtype=np.float64),
            accepted_grain=np.asarray(history["accepted_grain"],
                                      dtype=np.intp),
            rotation_amplitude_deg=np.asarray(
                history["rotation_amplitude_deg"], dtype=np.float64,
            ),
            amplitude_phase=np.asarray(
                history["amplitude_phase"], dtype=np.intp,
            ),
            pass_index=np.asarray(
                history["amplitude_phase"], dtype=np.intp,
            ),
            trajectory=traj,
        )
        self.refine_grains_history = history_out
        self.current_distribution = None
        if hasattr(self, "_rebuild_spatial_index"):
            self._rebuild_spatial_index()
        return history_out

    # =====================================================================
    # refine_initial_orientations — build-time SO(3) coordinate descent
    # =====================================================================
    #
    # Goal: choose each grain's rotation BEFORE the global FIRE quench so
    # that ``generate()`` lands in a deeper basin to begin with.  The big
    # win over post-hoc ``refine_grains*`` methods is that we don't need
    # FIRE per trial — the cell hasn't been relaxed yet, so trial scoring
    # uses *local* energy with a CACHED bond/angle/repulsion topology.
    # Each trial is ~1 ms instead of ~140 ms, so we can afford 50+ trials
    # per (amplitude × grain) and converge on good rotations fast.
    #
    # Design summary:
    #   1.  Build the topology ONCE on the as-built (un-relaxed) cell.
    #   2.  Coarse-to-fine amplitude schedule (default 30° → 15° → 5° →
    #       2°).  For each (amplitude, round, grain):
    #         - sample N bounded rotation perturbations of the grain's
    #           current rotation,
    #         - retile the grain with the spherical master block,
    #         - score by ``_total_energy_fast`` using the cached topo
    #           (sub-millisecond),
    #         - pick the best, commit if it improves on the baseline.
    #   3.  Rebuild topology between amplitude phases so it stays
    #       accurate as positions evolve.
    #
    # Why it generalises across chemistry: the energy kernel and topology
    # builder both consume per-species targets from ``shell_target``
    # (bond peak, angle mode, hard-core, non-bonded push), so Cu (1
    # species), Si (1), SiO₂ (2), and SrTiO₃ (3) all use the same code
    # path.  No per-element heuristics.
    #
    # Why "cached topology" is fair: the bonds we care about are the
    # ones predicted by the IDEAL crystal target — those don't depend on
    # where the atoms drifted to, only on which atoms got assigned to
    # the same coordination shell.  Local lattice perturbations don't
    # break this assignment.  When the topology drifts (after many
    # rotations), we rebuild between amplitude rounds.
    def refine_initial_orientations(
        self: "Supercell",
        shell_target: "CoordinationShellTarget",
        *,
        amplitudes_deg: tuple = (30.0, 15.0, 5.0, 2.0),
        trials_per_amplitude_per_grain: int = 50,
        max_rounds_per_amplitude: int = 2,
        # Spring weights (used to score trials)
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 3.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        time_budget_sec: float = 120.0,
        # Final whole-cell FIRE refinement after coordinate descent.
        # ``0`` (default) skips it — the user is expected to run
        # their own ``shell_relax`` afterwards (or
        # ``generate(refine_orientations=True)`` does it
        # automatically).  Set > 0 to bake the FIRE quench into the
        # refinement call so a single ``refine_initial_orientations``
        # produces a fully-relaxed cell.
        final_fire_steps: int = 0,
        # Capture per-frame atom positions for trajectory replay /
        # cost-decomposition.  ``False`` (default) skips this for
        # speed.
        capture_trajectory: bool = False,
        # Cost-function modes:
        #   ``"pair_distance"`` (default, recommended) — topology-free
        #     local pair-distance cost ``(d - pair_peak)²`` over grain
        #     atoms within ``score_cutoff_factor × pair_peak``, plus a
        #     hard-core clash penalty.  Fast, position-only, no cached
        #     bond list, so the score is consistent across grains and
        #     trials.  Generalises to any chemistry (Cu, Si, SiO₂,
        #     SrTiO₃) via per-pair targets in ``shell_target``.
        #   ``"cached_topology"`` — uses ``_total_energy_fast`` with a
        #     bond list rebuilt at the cadence below.  More physically
        #     faithful (includes angles, repulsion) but the rebuilt
        #     topology drifts as positions change, which can cause the
        #     refinement to walk into a worse basin on big cells.
        cost_function: str = "pair_distance",
        score_cutoff_factor: float = 1.5,
        # Topology-rebuild cadence (only used when
        # ``cost_function="cached_topology"``).
        topology_rebuild: str = "per_grain",
        rng_seed: "int | None" = None,
        show_progress: bool = True,
    ) -> dict:
        """Optimise per-grain rotations via SO(3) coordinate descent BEFORE the global FIRE quench.

        Walks each Voronoi grain through a sequence of progressively
        finer rotation perturbations, accepting any rotation that
        lowers a fast topology-free pair-distance score against the
        grain's local environment.  Designed to be called between
        the Voronoi tile (which assigns random initial rotations) and
        the global FIRE quench.  The intended workflow is::

            cell.generate(shell, num_steps=0, ...)        # build only
            cell.refine_initial_orientations(shell)        # this method
            cell.shell_relax(shell, num_steps=150, ...)    # FIRE quench

        Or equivalently use ``cell.generate(refine_orientations=True,
        refine_orientations_kwargs=...)`` which chains the three
        steps in one call.

        Parameters
        ----------
        shell_target : CoordinationShellTarget
            Target whose ``pair_peak`` defines the per-pair bond
            length the score targets.
        amplitudes_deg : tuple of float, optional
            Schedule of rotation amplitudes (degrees) the SO(3)
            coordinate search walks through.  Default
            ``(30, 15, 5, 2)``: the largest step lets a misaligned
            grain escape its starting basin, the smallest step locks
            in the chosen orientation.
        trials_per_amplitude_per_grain : int, optional
            Number of random rotations sampled per (amplitude,
            grain).  Default ``50``.  The best-scoring trial is
            accepted if it beats the current orientation by more
            than ``score_cutoff_factor``.
        max_rounds_per_amplitude : int, optional
            Number of full passes over all grains within one
            amplitude phase.  Default ``2``.
        bond_weight, angle_weight, repulsion_weight : float, optional
            Spring weights forwarded to the per-trial score.  Only
            used when ``cost_function="cached_topology"``; the
            default ``"pair_distance"`` mode ignores them.
        hard_core_scale, nonbond_push_scale : float, optional
            Repulsion thresholds passed through to the per-trial
            score's clash-penalty term.
        time_budget_sec : float, optional
            Wall-time guard rail (seconds).  The search bails after
            this even if amplitudes remain.  Default ``120``.
        final_fire_steps : int, optional
            If > 0, run a whole-cell ``shell_relax`` for this many
            steps after the SO(3) search completes — bakes a final
            quench into a single call.  Default ``0`` (caller is
            expected to run their own ``shell_relax``).
        capture_trajectory : bool, optional
            Record the cell's atom positions at every accepted
            rotation.  Default ``False`` (faster).  Set to ``True``
            for trajectory-replay HTML export.
        cost_function : {"pair_distance", "cached_topology"}, optional
            Score to minimise.  ``"pair_distance"`` (default,
            recommended) is a topology-free
            ``Σ (d - pair_peak)²`` over neighbour pairs in the
            grain's local frame — fast and consistent across grains
            and trials.  ``"cached_topology"`` uses
            ``_total_energy_fast`` with a rebuilt bond list (more
            physically faithful but the rebuilt topology drifts as
            positions change and can walk into a worse basin on big
            cells).
        score_cutoff_factor : float, optional
            Acceptance threshold relative to the current baseline
            score.  Higher values accept more aggressively.  Default
            ``1.5``.
        topology_rebuild : {"per_grain", "per_amp", "once"}, optional
            Cadence for rebuilding the bond list (only used when
            ``cost_function="cached_topology"``).  Default
            ``"per_grain"``.
        rng_seed : int, optional
            Seed for the random rotation sampler.  ``None`` (default)
            uses the cell's own RNG.
        show_progress : bool, optional
            Display a tqdm progress bar over the (amplitudes ×
            rounds × grains) workload.  Default ``True``.

        Returns
        -------
        dict
            History captured under
            ``self.refine_initial_orientations_history``:

            - ``iteration`` (ndarray of int) — accept indices,
              starting at 0 for the initial state.
            - ``global_cost`` (ndarray of float) — total cost at
              each accepted state.
            - ``cost_bond`` / ``cost_angle`` / ``cost_rep`` — cost
              decomposition (only populated for
              ``cost_function="cached_topology"``).
            - ``accepted_grain`` (ndarray of int) — grain index
              that moved at each acceptance (-1 for the initial
              state).
            - ``rotation_amplitude_deg`` (ndarray of float) — the
              current amplitude phase at each acceptance.
            - ``amplitude_phase`` (ndarray of int) — phase index
              into ``amplitudes_deg``.
            - ``trajectory`` (ndarray of float32, optional) — only
              present when ``capture_trajectory=True``: positions
              ``(num_accepts, num_atoms, 3)`` at each accepted
              rotation.
        """
        from ._thermal_mc import _build_thermal_topology, _total_energy_fast

        if getattr(self, "_grain_ids", None) is None:
            raise ValueError(
                "refine_initial_orientations requires a grain-built "
                "cell.  Call Supercell.generate(grain_size=...) first."
            )
        if getattr(self, "_grain_cells", None) is None:
            raise ValueError(
                "refine_initial_orientations requires Voronoi cells "
                "cached on the supercell.  Re-run generate() to "
                "populate them."
            )

        rng = (np.random.default_rng(rng_seed)
               if rng_seed is not None else self.rng)

        weights = dict(
            bond_weight=float(bond_weight),
            angle_weight=float(angle_weight),
            repulsion_weight=float(repulsion_weight),
            hard_core_scale=float(hard_core_scale),
            nonbond_push_scale=float(nonbond_push_scale),
        )

        # Cell + grain metadata
        grain_ids = np.asarray(self._grain_ids, dtype=np.intp)
        grain_seeds = np.asarray(self._grain_seeds, dtype=np.float64)
        voronoi_cells = self._grain_cells
        masters = self._grain_masters
        grain_source = self._grain_source
        if grain_source is None:
            grain_source = np.zeros(len(grain_seeds), dtype=np.intp)
        is_crystalline = np.asarray(
            self._grain_is_crystalline, dtype=bool,
        )
        box_dim = np.asarray(self._grain_box_dim, dtype=np.float64)
        master_lattice = np.asarray(
            self._grain_master_lattice, dtype=np.float64,
        )
        translation_basis = master_lattice * 0.5

        cell_mat = np.ascontiguousarray(
            self.atoms.cell.array, dtype=np.float64,
        )
        cell_inv = np.linalg.inv(cell_mat)
        species_idx = (
            self._atom_shell_species_index
            if getattr(self, "_atom_shell_species_index", None) is not None
            else self._atom_species_index
        ).astype(np.intp, copy=True)
        num_atoms = len(self.atoms)

        unique_grains = [
            int(g) for g in np.unique(grain_ids[grain_ids >= 0])
            if is_crystalline[int(g)]
        ]
        if not unique_grains:
            raise ValueError(
                "No crystalline grains found.  "
                "refine_initial_orientations has nothing to optimise."
            )

        # Per-grain orientation state (start from generate's choices)
        max_grain_id = int(grain_ids.max()) + 1
        current_rotations = np.zeros((max_grain_id, 3, 3),
                                     dtype=np.float64)
        for g in range(max_grain_id):
            current_rotations[g] = (
                self._grain_rotations_initial[g]
                if g < len(self._grain_rotations_initial)
                else np.eye(3)
            )
        current_translations = np.zeros((max_grain_id, 3),
                                        dtype=np.float64)

        # Per-pair targets (used by both cost-function modes).
        pair_peak_arr = np.asarray(
            shell_target.pair_peak, dtype=np.float64,
        )
        pair_outer_arr = np.asarray(
            shell_target.pair_outer, dtype=np.float64,
        )
        pair_hard_arr = np.asarray(
            shell_target.pair_hard_min, dtype=np.float64,
        )
        # Cutoff = ``score_cutoff_factor × max(pair_peak)``.  Using
        # the LARGEST pair-peak gives a wide enough cutoff to capture
        # boundary pairs of every species combination (Si-O, O-O,
        # Si-Si in SiO₂; Sr-Ti, Sr-O, Ti-O in SrTiO₃) — those are the
        # pairs whose distances actually CHANGE under grain rotation
        # (intra-grain pairs are rigid).  A tighter cutoff loses the
        # rotation signal entirely on multi-element cells.
        score_cutoff = (
            float(score_cutoff_factor)
            * float(np.max(pair_peak_arr))
        )

        # ── Build topology cache + energy kernel (only used when
        # ``cost_function="cached_topology"``).  For the default
        # ``"pair_distance"`` mode we never call _build_thermal_topology.
        def _build_topo():
            return _build_thermal_topology(
                self.atoms, species_idx, shell_target,
                hard_core_scale=float(weights["hard_core_scale"]),
                nonbond_push_scale=float(weights["nonbond_push_scale"]),
            )

        topo = _build_topo() if cost_function == "cached_topology" else None

        # Score function for a per-grain trial.  Two modes:
        #   * "pair_distance": topology-free, position-only, restricted
        #     to the grain's neighbourhood.  Consistent across grains.
        #   * "cached_topology": uses _total_energy_fast with the
        #     current cached topology.  Includes angles + repulsion.
        def _score(positions, grain_mask, neighborhood_mask):
            if cost_function == "pair_distance":
                return _pair_distance_cost(
                    positions, species_idx, cell_mat, cell_inv,
                    pair_peak_arr, pair_hard_arr, score_cutoff,
                    grain_mask=grain_mask,
                    neighborhood_mask=neighborhood_mask,
                )
            r_dummy = np.zeros_like(positions)
            total, _, _, _ = _total_energy_fast(
                positions, species_idx, cell_mat, cell_inv,
                topo["bond_i"], topo["bond_j"], topo["bond_r_target"],
                float(weights["bond_weight"]),
                topo["tri_center"], topo["tri_a"], topo["tri_b"],
                topo["tri_phi_target"],
                float(weights["angle_weight"]),
                topo["rep_atom_start"], topo["rep_atom_list"],
                topo["hard_core"], topo["nonbond_push"],
                float(weights["repulsion_weight"]),
                topo["bonded_flat"], num_atoms,
                r_dummy, 0.0,
            )
            return float(total) / max(num_atoms, 1)

        # ── History
        history = dict(
            iteration=[], global_cost=[],
            accepted_grain=[],
            amplitude_deg=[],
            phase_idx=[],
            trajectory=[] if capture_trajectory else None,
        )

        # Initial global score (full-cell pair-distance cost, both
        # origin and target unrestricted).
        initial_cost = _score(self.atoms.positions, None, None)
        history["iteration"].append(0)
        history["global_cost"].append(initial_cost)
        history["accepted_grain"].append(-1)
        history["amplitude_deg"].append(0.0)
        history["phase_idx"].append(-1)
        if capture_trajectory:
            history["trajectory"].append(
                self.atoms.positions.copy().astype(np.float32)
            )

        if show_progress:
            print(
                "refine_initial_orientations: initial cost = "
                f"{initial_cost:.4f}, {len(unique_grains)} grains, "
                f"amplitudes {amplitudes_deg}"
            )

        t0 = time.time()
        iteration = 0

        # ── Main loop
        for amp_idx, amp_deg in enumerate(amplitudes_deg):
            if time.time() - t0 >= time_budget_sec:
                break
            amp_rad = np.deg2rad(float(amp_deg))
            trans_scale = float(amp_deg / 180.0)

            for round_idx in range(max_rounds_per_amplitude):
                if time.time() - t0 >= time_budget_sec:
                    break
                grain_order = list(unique_grains)
                rng.shuffle(grain_order)
                improved_round = False

                for gid in grain_order:
                    if time.time() - t0 >= time_budget_sec:
                        break

                    grain_mask = (grain_ids == gid)
                    target_n = int(np.sum(grain_mask))
                    if target_n == 0:
                        continue
                    seed_world = grain_seeds[gid]
                    voronoi_cell = voronoi_cells[gid]
                    src_idx = int(grain_source[gid])
                    master = masters[src_idx]
                    master_pos = np.asarray(
                        master["positions"], dtype=np.float64,
                    )
                    master_num = np.asarray(
                        master["numbers"], dtype=np.int64,
                    )

                    snapshot_grain_pos = self.atoms.positions[grain_mask].copy()
                    # Multi-species: retile re-orders atoms within the
                    # grain (same species count, different positions).
                    # We must update ``atoms.numbers`` AND
                    # ``species_idx`` along with positions so the cost
                    # function sees the correct species at each
                    # position.  Without this the score is computed
                    # against a stale species mapping and refinement
                    # fails on multi-element cells (Si–O, Sr–Ti–O).
                    snapshot_grain_nums = self.atoms.numbers[grain_mask].copy()
                    grain_indices = np.flatnonzero(grain_mask)
                    snapshot_species_idx_grain = species_idx[grain_indices].copy()
                    # Pre-compute neighbourhood mask for the local
                    # pair-distance score.  Restricts j-atoms to the
                    # grain + atoms within ``score_cutoff`` of any
                    # grain atom, so each trial costs O(N_g · N_n)
                    # instead of O(N²).
                    neighborhood_mask = _expand_neighborhood(
                        grain_mask=grain_mask,
                        positions=self.atoms.positions,
                        cell_mat=cell_mat, cell_inv=cell_inv,
                        radius=score_cutoff,
                    )
                    if (cost_function == "cached_topology"
                            and topology_rebuild == "per_grain"):
                        topo = _build_topo()
                    baseline_cost = _score(
                        self.atoms.positions, grain_mask, neighborhood_mask,
                    )
                    best_cost = baseline_cost
                    best_R = None
                    best_T = None
                    best_atoms_grain: "np.ndarray | None" = None
                    best_nums_grain: "np.ndarray | None" = None

                    for _ in range(trials_per_amplitude_per_grain):
                        R_delta = _so3_bounded_rotation(rng, amp_rad)
                        R_trial = current_rotations[gid] @ R_delta
                        frac = rng.uniform(-1.0, 1.0, size=3)
                        T_trial = (current_translations[gid]
                                   + trans_scale * (frac @ translation_basis))

                        retile = _retile_grain(
                            master_positions=master_pos,
                            master_numbers=master_num,
                            voronoi_cell=voronoi_cell,
                            seed_world=seed_world,
                            box_dim=box_dim,
                            rotation=R_trial,
                            translation=T_trial,
                            target_n=target_n,
                        )
                        if retile is None:
                            continue
                        new_pos_world, new_nums_world = retile
                        self.atoms.positions[grain_mask] = new_pos_world
                        self.atoms.numbers[grain_mask] = new_nums_world
                        # Re-map species index for the grain atoms
                        species_idx[grain_indices] = np.searchsorted(
                            self._species, new_nums_world,
                        )
                        if (cost_function == "cached_topology"
                                and topology_rebuild == "per_trial"):
                            topo = _build_topo()
                        cost = _score(
                            self.atoms.positions, grain_mask,
                            neighborhood_mask,
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_R = R_trial
                            best_T = T_trial
                            best_atoms_grain = new_pos_world.copy()
                            best_nums_grain = new_nums_world.copy()

                    # Restore snapshot first, then either commit best
                    # or stay at baseline.
                    self.atoms.positions[grain_mask] = snapshot_grain_pos
                    self.atoms.numbers[grain_mask] = snapshot_grain_nums
                    species_idx[grain_indices] = snapshot_species_idx_grain
                    if best_atoms_grain is not None:
                        self.atoms.positions[grain_mask] = best_atoms_grain
                        self.atoms.numbers[grain_mask] = best_nums_grain
                        species_idx[grain_indices] = np.searchsorted(
                            self._species, best_nums_grain,
                        )
                        current_rotations[gid] = best_R
                        current_translations[gid] = best_T
                        iteration += 1
                        history["iteration"].append(iteration)
                        history["global_cost"].append(best_cost)
                        history["accepted_grain"].append(int(gid))
                        history["amplitude_deg"].append(float(amp_deg))
                        history["phase_idx"].append(amp_idx)
                        if capture_trajectory:
                            history["trajectory"].append(
                                self.atoms.positions.copy()
                                .astype(np.float32)
                            )
                        improved_round = True
                        if show_progress:
                            elapsed = time.time() - t0
                            print(
                                f"  amp {amp_deg:5.1f}° "
                                f"round {round_idx+1} grain {gid}: "
                                f"{baseline_cost:.4f} -> {best_cost:.4f} "
                                f"(iter {iteration}, "
                                f"elapsed {elapsed:.1f} s)"
                            )

                if not improved_round:
                    if show_progress:
                        print(
                            f"  amp {amp_deg:5.1f}° round {round_idx+1}:"
                            " no improvements"
                        )
                    break

            # Rebuild topology between amplitudes — positions changed
            if (cost_function == "cached_topology"
                    and topology_rebuild == "per_amplitude"
                    and amp_idx < len(amplitudes_deg) - 1):
                topo = _build_topo()

        final_cost = _score(self.atoms.positions, None, None)
        if show_progress:
            print(
                f"refine_initial_orientations: refine cost = "
                f"{final_cost:.4f} (Δ = {final_cost - initial_cost:+.4f})"
            )
            print(f"  total accepts: {iteration}")
            print(f"  refine time: {time.time() - t0:.1f} s")

        # Optional final FIRE refinement of the rotated state.  This
        # is the "FIRE refinement at the end" step — the rotation
        # search produced grain orientations that look good by the
        # cheap pair-distance metric, and the final FIRE relaxes all
        # atoms (especially boundary atoms that the rotation search
        # held rigid) to a deeper basin.  Required for the algorithm
        # to actually lower the post-quench global cost.
        if final_fire_steps > 0:
            if show_progress:
                print(
                    f"refine_initial_orientations: final FIRE "
                    f"({final_fire_steps} steps)"
                )
            t_fire = time.time()
            self.shell_relax(
                shell_target,
                num_steps=int(final_fire_steps),
                neighbor_update_interval=99999,
                capture_trajectory=False,
                show_progress=False,
                **{k: v for k, v in weights.items()
                   if k in ("bond_weight", "angle_weight",
                            "repulsion_weight", "hard_core_scale",
                            "nonbond_push_scale")},
            )
            if capture_trajectory:
                history["iteration"].append(iteration + 1)
                history["global_cost"].append(
                    _score(self.atoms.positions, None, None)
                )
                history["accepted_grain"].append(-2)  # final-quench sentinel
                history["amplitude_deg"].append(0.0)
                history["phase_idx"].append(-1)
                history["trajectory"].append(
                    self.atoms.positions.copy().astype(np.float32)
                )
            if show_progress:
                print(f"  final FIRE time: {time.time() - t_fire:.1f} s")

        # Persist optimised rotations on the cell so subsequent
        # refine_grains*-style methods know the new orientations.
        for gid in unique_grains:
            self._grain_rotations_initial[gid] = current_rotations[gid]

        # Re-sync the cell's species-index caches with the current
        # ``atoms.numbers`` (which may have changed if multi-species
        # grains rotated and re-ordered Si/O slots).  Without this
        # the subsequent FIRE quench would see a stale mapping.
        self._atom_species_index = np.searchsorted(
            self._species, self.atoms.numbers,
        )
        if getattr(self, "_atom_shell_species_index", None) is not None:
            # The shell-species index follows the same mapping when
            # we don't have explicit per-atom virtual species.
            self._atom_shell_species_index = species_idx.copy()

        if hasattr(self, "_rebuild_spatial_index"):
            self._rebuild_spatial_index()

        traj_arr = (
            np.asarray(history["trajectory"], dtype=np.float32)
            if capture_trajectory else None
        )
        history_out = dict(
            iteration=np.asarray(history["iteration"], dtype=np.intp),
            global_cost=np.asarray(history["global_cost"],
                                   dtype=np.float64),
            accepted_grain=np.asarray(history["accepted_grain"],
                                      dtype=np.intp),
            amplitude_deg=np.asarray(history["amplitude_deg"],
                                     dtype=np.float64),
            phase_idx=np.asarray(history["phase_idx"], dtype=np.intp),
            trajectory=traj_arr,
        )
        self.refine_initial_orientations_history = history_out
        return history_out
