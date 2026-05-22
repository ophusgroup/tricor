"""Numba JIT kernel for the FIRE topology-rebuild bond-matching loop.

The Python loop in ``_shell_relax.py:rebuild_topology`` iterates over
~6 M neighbour pairs at 200³ Å × 608 k atoms, doing hash-set lookups +
list-of-list bookkeeping for each pair.  That loop is the single
slowest piece of the FIRE pipeline at scale.  This module replaces it
with a JIT'd version that uses flat numpy arrays for everything.

The kernel produces bit-equivalent output (the same set of accepted
bonds, in the same order) as the Python reference, so it's safe to
drop in.  The caller passes ``nl_i, nl_j, nl_d, nl_hats`` already
sorted by distance.

Output bookkeeping:
- ``bond_i, bond_j, bond_r`` — accepted bonds (1-D arrays)
- ``bonded_nbr (num_atoms, max_k)`` — per-atom bonded neighbour indices,
  filled left-to-right; ``-1`` marks empty slots.  Derive
  ``bonded_neighbors`` (list-of-list) and ``bonded_set`` from this in
  Python after the JIT call.
- ``bond_count (num_atoms,)`` — current bond count per atom.

The Python fallback path stays available - if numba is unavailable or
the cache is busted, the caller catches ``ImportError`` and runs the
old loop.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


if HAS_NUMBA:

    @njit(cache=True, fastmath=False)
    def _build_bond_graph(
        nl_i: np.ndarray,           # (P,) int64  - sorted by distance
        nl_j: np.ndarray,           # (P,) int64
        nl_hats: np.ndarray,        # (P, 3) float64  - sorted, unit vectors
        species_idx: np.ndarray,    # (N,) int64
        k_atom: np.ndarray,         # (N,) int64  - per-atom coordination cap
        coord_target_int: np.ndarray,  # (S, S) int64
        pair_peak: np.ndarray,      # (S, S) float64
        max_k: int,                 # int  - max per-atom bonds (bounds bonded_nbr)
        num_atoms: int,
        num_sp: int,
        cos_thresh: float,          # cos(min_accept_angle), e.g. cos(60°)=0.5
    ):
        """Greedy bond matching with two passes (angle-aware, then distance-only).

        Mirrors the Python rebuild_topology body exactly.  See module
        docstring for output layout.
        """
        # Per-atom state.  bonded_nbr packs neighbour indices left-to-right
        # in row ``i``; bond_count[i] is the current fill level.
        bond_count = np.zeros(num_atoms, dtype=np.int64)
        bond_count_pair = np.zeros((num_atoms, num_sp), dtype=np.int64)
        bonded_nbr = np.full((num_atoms, max_k), -1, dtype=np.int64)
        bond_hats = np.zeros((num_atoms, max_k, 3), dtype=np.float64)

        n_pairs = nl_i.shape[0]
        # Upper bound on accepted bonds: every pair could in principle
        # be accepted (won't happen, but the arrays are cheap).
        bond_i = np.empty(n_pairs, dtype=np.int64)
        bond_j = np.empty(n_pairs, dtype=np.int64)
        bond_r = np.empty(n_pairs, dtype=np.float64)
        n_bonds = 0

        # --- pass 1: angle-aware ---
        for k in range(n_pairs):
            ai = nl_i[k]
            aj = nl_j[k]
            if bond_count[ai] >= k_atom[ai] or bond_count[aj] >= k_atom[aj]:
                continue
            si = species_idx[ai]
            sj = species_idx[aj]
            if bond_count_pair[ai, sj] >= coord_target_int[si, sj]:
                continue
            if bond_count_pair[aj, si] >= coord_target_int[sj, si]:
                continue
            # Already bonded?  Scan the (small) bonded_nbr row.
            n_ai = bond_count[ai]
            already = False
            for ii in range(n_ai):
                if bonded_nbr[ai, ii] == aj:
                    already = True
                    break
            if already:
                continue
            # Angle check at ai: hat_ij vs each existing hat.
            hx = nl_hats[k, 0]
            hy = nl_hats[k, 1]
            hz = nl_hats[k, 2]
            ok = True
            for ii in range(n_ai):
                ex = bond_hats[ai, ii, 0]
                ey = bond_hats[ai, ii, 1]
                ez = bond_hats[ai, ii, 2]
                if hx * ex + hy * ey + hz * ez > cos_thresh:
                    ok = False
                    break
            if not ok:
                continue
            # Angle check at aj with reversed vector hat_ji = -hat_ij.
            n_aj = bond_count[aj]
            for jj in range(n_aj):
                ex = bond_hats[aj, jj, 0]
                ey = bond_hats[aj, jj, 1]
                ez = bond_hats[aj, jj, 2]
                if (-hx) * ex + (-hy) * ey + (-hz) * ez > cos_thresh:
                    ok = False
                    break
            if not ok:
                continue
            # Accept.
            bonded_nbr[ai, n_ai] = aj
            bonded_nbr[aj, n_aj] = ai
            bond_hats[ai, n_ai, 0] = hx
            bond_hats[ai, n_ai, 1] = hy
            bond_hats[ai, n_ai, 2] = hz
            bond_hats[aj, n_aj, 0] = -hx
            bond_hats[aj, n_aj, 1] = -hy
            bond_hats[aj, n_aj, 2] = -hz
            bond_count[ai] = n_ai + 1
            bond_count[aj] = n_aj + 1
            bond_count_pair[ai, sj] += 1
            bond_count_pair[aj, si] += 1
            bond_i[n_bonds] = ai
            bond_j[n_bonds] = aj
            bond_r[n_bonds] = pair_peak[si, sj]
            n_bonds += 1

        # --- pass 2: distance-only ---
        for k in range(n_pairs):
            ai = nl_i[k]
            aj = nl_j[k]
            if bond_count[ai] >= k_atom[ai] or bond_count[aj] >= k_atom[aj]:
                continue
            si = species_idx[ai]
            sj = species_idx[aj]
            if bond_count_pair[ai, sj] >= coord_target_int[si, sj]:
                continue
            if bond_count_pair[aj, si] >= coord_target_int[sj, si]:
                continue
            n_ai = bond_count[ai]
            already = False
            for ii in range(n_ai):
                if bonded_nbr[ai, ii] == aj:
                    already = True
                    break
            if already:
                continue
            n_aj = bond_count[aj]
            bonded_nbr[ai, n_ai] = aj
            bonded_nbr[aj, n_aj] = ai
            bond_count[ai] = n_ai + 1
            bond_count[aj] = n_aj + 1
            bond_count_pair[ai, sj] += 1
            bond_count_pair[aj, si] += 1
            bond_i[n_bonds] = ai
            bond_j[n_bonds] = aj
            bond_r[n_bonds] = pair_peak[si, sj]
            n_bonds += 1

        return (
            bond_i[:n_bonds].copy(),
            bond_j[:n_bonds].copy(),
            bond_r[:n_bonds].copy(),
            bonded_nbr,
            bond_count,
        )

else:  # pragma: no cover
    _build_bond_graph = None


def build_bond_graph_numba(
    nl_i_sorted: np.ndarray,
    nl_j_sorted: np.ndarray,
    nl_hats_sorted: np.ndarray,
    species_idx: np.ndarray,
    k_atom: np.ndarray,
    coord_target_int: np.ndarray,
    pair_peak: np.ndarray,
    num_atoms: int,
    num_sp: int,
    min_accept_angle_rad: float,
):
    """Friendly wrapper around the JIT kernel.

    Casts inputs to int64 / float64, computes ``max_k`` from
    ``k_atom``, and forwards to the JIT'd ``_build_bond_graph``.
    Raises ``RuntimeError`` if numba is unavailable so the caller
    knows to fall back.
    """
    if not HAS_NUMBA or _build_bond_graph is None:
        raise RuntimeError(
            "numba is not available; build_bond_graph_numba cannot run.",
        )
    nl_i = np.ascontiguousarray(nl_i_sorted, dtype=np.int64)
    nl_j = np.ascontiguousarray(nl_j_sorted, dtype=np.int64)
    nl_hats = np.ascontiguousarray(nl_hats_sorted, dtype=np.float64)
    sp = np.ascontiguousarray(species_idx, dtype=np.int64)
    k_arr = np.ascontiguousarray(k_atom, dtype=np.int64)
    ctgt = np.ascontiguousarray(coord_target_int, dtype=np.int64)
    pp = np.ascontiguousarray(pair_peak, dtype=np.float64)
    max_k = int(k_arr.max()) if k_arr.size else 0
    if max_k == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.full((num_atoms, 0), -1, dtype=np.int64),
            np.zeros(num_atoms, dtype=np.int64),
        )
    cos_thresh = float(np.cos(float(min_accept_angle_rad)))
    return _build_bond_graph(
        nl_i, nl_j, nl_hats,
        sp, k_arr, ctgt, pp,
        max_k, int(num_atoms), int(num_sp),
        cos_thresh,
    )
