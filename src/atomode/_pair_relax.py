"""Pure-geometric pair relaxation helpers (no torch / ML dependency).

Provides two ``scipy.spatial.cKDTree``-based sweeps used by
:meth:`Supercell.bond_relax` and :meth:`Supercell.enforce_hard_core`:

- :func:`_bond_relax_sweep` - combined attract-to-bond-peak +
  repel-from-hard-core, with a per-atom displacement cap.  The
  workhorse behind the shortcut pipeline used in NB 01 / NB 02.
- :func:`_enforce_hard_core` - pure geometric projection step that
  pushes any pair below the hard-core wall apart by ``push_fraction``
  of the deficit.  Useful as an opt-in cleanup after FIRE in dense
  regions where bond springs can pull pairs through their walls.

Both functions assume orthorhombic periodic cells (the ``boxsize``
argument to ``cKDTree``) and full PBC.  Single-thread, vectorised at
the pair level via numpy.
"""

from __future__ import annotations

import numpy as np


def _bond_relax_sweep(
    positions: np.ndarray,
    box: np.ndarray,
    species_idx: np.ndarray,
    pair_peak: np.ndarray,
    pair_hard_min: np.ndarray,
    pair_outer: np.ndarray,
    coordination_target: np.ndarray,
    n_iter: int = 40,
    attract_frac: float = 0.2,
    repel_frac: float = 1.0,
    max_step: float = 0.2,
) -> np.ndarray:
    """Combined attract-to-bond-peak + repel-from-hard-core sweep.

    Faster than a full FIRE quench for the Voronoi-tile → bond-correct
    transition:

    - Each sweep is one ``cKDTree`` query at O(N log N), not a full
      FIRE step with neighbour-list rebuilds.  ~1-2 s/sweep at
      200³ Å × 600 k atoms.
    - Overlap-aware by construction: bond springs cannot pull pairs
      through the hard core in tight regions (the FIRE failure mode).
    - Drives the bonded pair to its actual peak rather than parking
      atoms at the hard-core wall.

    Per pair within ``max_cutoff``:

    - If ``coordination_target[s_i, s_j] > 0`` AND
      ``dist <= pair_outer[s_i, s_j]``:
      pull toward ``pair_peak[s_i, s_j]`` at rate ``attract_frac``.
    - If ``dist < pair_hard_min[s_i, s_j]``:
      push apart at rate ``repel_frac``.

    Per-atom step is clipped to ``max_step`` Å to prevent runaway
    motion in dense regions.

    Parameters
    ----------
    positions
        (N, 3) atom positions in Å.  Modified in place.
    box
        (3,) orthorhombic box side lengths.
    species_idx
        (N,) per-atom species index.
    pair_peak, pair_hard_min, pair_outer
        (S, S) per-species-pair matrices from
        ``CoordinationShellTarget``.
    coordination_target
        (S, S) per-species-pair NN count.  Non-zero entries identify
        the *bonded* pairs that should be pulled to their peak.
    n_iter
        Number of sweeps.  20 typically reaches the bond peak to
        within 0.01 Å.
    attract_frac, repel_frac
        Per-sweep gap-closing fractions.  Attract is slower than
        repel to avoid overshooting through the bond peak.
    max_step
        Per-atom displacement cap (Å) per sweep.
    """
    from scipy.spatial import cKDTree

    box_np = np.asarray(box, dtype=np.float64)
    sp = np.asarray(species_idx)
    pp = np.asarray(pair_peak, dtype=np.float64)
    phc = np.asarray(pair_hard_min, dtype=np.float64)
    po = np.asarray(pair_outer, dtype=np.float64)
    bonded = np.asarray(coordination_target, dtype=np.float64) > 0
    pull_cut = float(po[bonded].max()) if bonded.any() else 0.0
    max_cut = max(pull_cut, float(phc.max())) * 1.05
    if max_cut <= 0.0:
        return positions
    pos = positions
    for _ in range(int(n_iter)):
        wrap = pos - np.floor(pos / box_np) * box_np
        tree = cKDTree(wrap, boxsize=box_np)
        pairs = tree.query_pairs(max_cut, output_type="ndarray")
        if not len(pairs):
            break
        si = sp[pairs[:, 0]]
        sj = sp[pairs[:, 1]]
        delta = wrap[pairs[:, 1]] - wrap[pairs[:, 0]]
        delta -= np.round(delta / box_np) * box_np
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-9)
        unit = delta / dist[:, None]

        hc = phc[si, sj]
        peak = pp[si, sj]
        outer = po[si, sj]
        is_bond = bonded[si, sj]

        attract = np.where(
            is_bond & (dist <= outer), (dist - peak) * attract_frac, 0.0)
        repel = np.where(dist < hc, -(hc - dist) * repel_frac, 0.0)
        signed = attract + repel

        disp = unit * signed[:, None]
        atom_disp = np.zeros_like(pos)
        np.add.at(atom_disp, pairs[:, 0], disp)
        np.add.at(atom_disp, pairs[:, 1], -disp)
        # Per-atom clipping after accumulation (handles atoms in dense
        # regions that receive many overlapping pair contributions).
        mag = np.linalg.norm(atom_disp, axis=1).clip(min=1e-9)
        scale = np.clip(max_step / mag, 0.0, 1.0)
        atom_disp *= scale[:, None]
        pos += atom_disp
    return pos


def _enforce_hard_core(
    positions: np.ndarray,
    box: np.ndarray,
    species_idx: np.ndarray,
    pair_hard_min: np.ndarray,
    n_iter: int = 2,
    push_fraction: float = 0.5,
) -> np.ndarray:
    """Push apart any pair below its species-pair hard-core distance.

    Iteratively: find offenders via ``cKDTree``, move each pair apart
    along their bond vector by ``push_fraction × deficit``.  Converges
    in O(log(initial_deficit / push_fraction)) iterations for moderate
    overlaps; dense clusters may need more.

    Vectorised at the pair level (no per-atom Python loop) and
    O(N log N) per iteration thanks to the KDTree.  At 200³ Å × 600 k
    atoms: ~3 s per iteration on CPU.

    Parameters
    ----------
    positions
        (N, 3) atom positions in Å.  Modified in place.
    box
        (3,) orthorhombic box side lengths.
    species_idx
        (N,) per-atom species index into ``pair_hard_min``.
    pair_hard_min
        (S, S) per-species-pair hard-core distance (Å).  Typically
        ``shell_target.pair_hard_min``.
    n_iter
        Maximum number of cleanup iterations.  Early-terminates when
        no real violations remain (a small FP-noise tolerance avoids
        chasing ULP-level drift indefinitely).
    push_fraction
        Per-iteration push amount as a fraction of the deficit.
        ``0.5`` is the natural choice (atoms meet in the middle);
        larger values can overshoot.

    Returns
    -------
    np.ndarray
        ``positions``, modified in place.
    """
    from scipy.spatial import cKDTree

    pair_hc = np.asarray(pair_hard_min, dtype=np.float64)
    box_np = np.asarray(box, dtype=np.float64)
    species_np = np.asarray(species_idx)
    pos = positions  # (mutated in place)
    max_cutoff = float(pair_hc.max()) * 1.05
    if max_cutoff <= 0.0:
        return pos
    # Tolerance on the violation check: pairs sitting at the wall can
    # drift below by FP-ULP noise after the cKDTree round-trip and
    # `np.add.at` accumulation.  Without slack the loop keeps chasing
    # imaginary violations indefinitely; with slack the early-termination
    # kicks in once real overlaps are cleared.
    _hc_eps = 1e-9
    for _ in range(int(n_iter)):
        wrapped = pos - np.floor(pos / box_np) * box_np
        tree = cKDTree(wrapped, boxsize=box_np)
        pairs = tree.query_pairs(max_cutoff, output_type="ndarray")
        if len(pairs) == 0:
            break
        si = species_np[pairs[:, 0]]
        sj = species_np[pairs[:, 1]]
        target = pair_hc[si, sj]
        delta = wrapped[pairs[:, 1]] - wrapped[pairs[:, 0]]
        delta -= np.round(delta / box_np) * box_np
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-9)
        mask = dist < target - _hc_eps
        if not mask.any():
            break
        unit = delta[mask] / dist[mask, None]
        push_amt = (target[mask] - dist[mask]) * float(push_fraction)
        push = unit * push_amt[:, None]
        np.add.at(pos, pairs[mask, 0], -push)
        np.add.at(pos, pairs[mask, 1], push)
    return pos
