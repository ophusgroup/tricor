"""Numba-parallel kernel for :meth:`G3Distribution.measure_g3`.

Strictly an acceleration of the pure-numpy implementation in
``g3.py`` - no algorithmic change.  Both paths must produce
bit-identical ``g3count`` / ``g2count`` arrays (they're pure integer
accumulators).
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


if HAS_NUMBA:

    @njit(parallel=True, cache=False, fastmath=False)
    def _g3_kernel(
        origin_xyz,           # (N_origin, 3)           float64
        origin_species_id,    # (N_origin,)              intp
        tile_xyz_packed,      # (N_tile_total, 3)       float64
        tile_species_start,   # (num_species + 1,)       intp   (cumulative)
        g3_lookup,            # (num_sp, num_sp, num_sp) intp
        r_max_sq,             # float64
        r_step,               # float64
        phi_step,             # float64
        num_r,                # int
        num_phi,              # int
        num_species,          # int
        num_triplets,         # int
        zero_tol,             # float64
        thread_g3,            # (T, num_triplets, num_r, num_r, num_phi) int64
        thread_g2,            # (T, num_species, num_species, num_r)     int64
    ):
        """Accumulate per-thread g3 / g2 buffers over all origin atoms.

        Caller reduces along axis 0 of the two output buffers outside
        numba (faster than an explicit 5D numba reduction loop).
        """
        num_origins = origin_xyz.shape[0]
        T = thread_g3.shape[0]

        # Per-thread scratch for neighbour-list build.  Worst case each
        # origin sees every tile atom; size accordingly once per thread.
        N_tile_max = tile_xyz_packed.shape[0]
        scratch_x = np.empty((T, N_tile_max), dtype=np.float64)
        scratch_y = np.empty((T, N_tile_max), dtype=np.float64)
        scratch_z = np.empty((T, N_tile_max), dtype=np.float64)
        # Store r² (not r).  Using ``sqrt(r_sq_j * r_sq_k)`` as the
        # denominator matches numpy's ``sqrt(r01_sq * r02_sq)`` bit-for-
        # bit; computing ``sqrt(r_sq)`` and squaring it back would drift.
        scratch_r2 = np.empty((T, N_tile_max), dtype=np.float64)
        scratch_rb = np.empty((T, N_tile_max), dtype=np.intp)
        # Per-species start / count into the packed scratch arrays.
        scratch_sp_start = np.empty((T, num_species + 1), dtype=np.intp)

        for origin_i in prange(num_origins):
            tid = numba.get_thread_id()
            ox = origin_xyz[origin_i, 0]
            oy = origin_xyz[origin_i, 1]
            oz = origin_xyz[origin_i, 2]
            s_origin = origin_species_id[origin_i]

            sx = scratch_x[tid]
            sy = scratch_y[tid]
            sz = scratch_z[tid]
            sr2 = scratch_r2[tid]
            sb = scratch_rb[tid]
            sstart = scratch_sp_start[tid]

            # Build per-species neighbour arrays, packed contiguously
            # with species offsets in ``sstart``.
            write_idx = 0
            sstart[0] = 0
            for ind_n in range(num_species):
                t_start = tile_species_start[ind_n]
                t_end = tile_species_start[ind_n + 1]
                for k in range(t_start, t_end):
                    dx = tile_xyz_packed[k, 0] - ox
                    dy = tile_xyz_packed[k, 1] - oy
                    dz = tile_xyz_packed[k, 2] - oz
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 <= zero_tol or r2 >= r_max_sq:
                        continue
                    r = np.sqrt(r2)
                    r_bin = int(np.floor(r / r_step))
                    if r_bin >= num_r:
                        continue
                    sx[write_idx] = dx
                    sy[write_idx] = dy
                    sz[write_idx] = dz
                    sr2[write_idx] = r2
                    sb[write_idx] = r_bin
                    # Accumulate g2 count per species pair.
                    thread_g2[tid, s_origin, ind_n, r_bin] += 1
                    write_idx += 1
                sstart[ind_n + 1] = write_idx

            # Triplet accumulation for every channel whose center matches
            # this origin's species.
            for ind_1 in range(num_species):
                a_start = sstart[ind_1]
                a_end = sstart[ind_1 + 1]
                if a_start == a_end:
                    continue
                for ind_2 in range(ind_1, num_species):
                    b_start = sstart[ind_2]
                    b_end = sstart[ind_2 + 1]
                    if b_start == b_end:
                        continue
                    triplet_idx = g3_lookup[s_origin, ind_1, ind_2]
                    if triplet_idx < 0:
                        continue

                    same_species = ind_1 == ind_2
                    # Loop over all (j, k) neighbour pairs.  For same
                    # species skip the j == k diagonal.  For different
                    # species do the symmetric (k, j) write as well.
                    for j in range(a_start, a_end):
                        jx = sx[j]; jy = sy[j]; jz = sz[j]
                        jr2 = sr2[j]; jb = sb[j]
                        for k in range(b_start, b_end):
                            if same_species and j == k:
                                continue
                            kx = sx[k]; ky = sy[k]; kz = sz[k]
                            kr2 = sr2[k]; kb = sb[k]
                            dot = jx * kx + jy * ky + jz * kz
                            # Match the numpy path's float order exactly:
                            # ``sqrt(r01_sq * r02_sq)`` instead of
                            # ``sqrt(r01_sq) * sqrt(r02_sq)``.  The two
                            # forms are algebraically identical but ULP-
                            # different at floating-point edges, which
                            # was flipping a handful of same-species
                            # triplets into adjacent phi bins.
                            denom = np.sqrt(jr2 * kr2)
                            cos_phi = dot / denom
                            if cos_phi > 1.0:
                                cos_phi = 1.0
                            elif cos_phi < -1.0:
                                cos_phi = -1.0
                            phi = np.arccos(cos_phi)
                            phi_bin = int(np.floor(phi / phi_step))
                            if phi_bin >= num_phi:
                                phi_bin = num_phi - 1
                            elif phi_bin < 0:
                                phi_bin = 0
                            thread_g3[tid, triplet_idx, jb, kb, phi_bin] += 1
                            if not same_species:
                                thread_g3[tid, triplet_idx, kb, jb, phi_bin] += 1


def pack_tile_by_species(
    tile_xyz: np.ndarray,
    tile_species: np.ndarray,
    species_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Group tile atoms by species in ``species_order`` order.

    Returns
    -------
    tile_xyz_packed : (N_tile, 3) float64
    tile_species_start : (num_species + 1,) intp, cumulative start
        indices so atoms of species `s` live at rows
        ``tile_species_start[s]:tile_species_start[s+1]``.
    """
    num_species = int(species_order.size)
    per_species = [
        tile_xyz[tile_species == z].astype(np.float64, copy=False)
        for z in species_order
    ]
    counts = np.array([a.shape[0] for a in per_species], dtype=np.intp)
    starts = np.zeros(num_species + 1, dtype=np.intp)
    starts[1:] = np.cumsum(counts)
    if per_species:
        packed = np.concatenate(per_species, axis=0)
    else:
        packed = np.empty((0, 3), dtype=np.float64)
    return packed, starts


def run_g3_numba(
    origin_xyz: np.ndarray,
    origin_species_index: np.ndarray,
    tile_xyz: np.ndarray,
    tile_species: np.ndarray,
    species: np.ndarray,
    g3_lookup: np.ndarray,
    r_max: float,
    r_step: float,
    phi_step: float,
    num_r: int,
    num_phi: int,
    num_species: int,
    num_triplets: int,
    zero_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the numba-parallel accumulation and return new output arrays."""
    if not HAS_NUMBA:  # pragma: no cover
        raise RuntimeError("numba is not available; install tricor[fast].")

    tile_packed, tile_starts = pack_tile_by_species(
        tile_xyz, tile_species, species,
    )
    origin_xyz = np.ascontiguousarray(origin_xyz, dtype=np.float64)
    origin_species_index = np.ascontiguousarray(origin_species_index, dtype=np.intp)
    g3_lookup = np.ascontiguousarray(g3_lookup, dtype=np.intp)

    T = int(numba.get_num_threads())
    thread_g3 = np.zeros(
        (T, num_triplets, num_r, num_r, num_phi), dtype=np.int64
    )
    thread_g2 = np.zeros(
        (T, num_species, num_species, num_r), dtype=np.int64
    )
    _g3_kernel(
        origin_xyz,
        origin_species_index,
        tile_packed,
        tile_starts,
        g3_lookup,
        float(r_max * r_max),
        float(r_step),
        float(phi_step),
        int(num_r),
        int(num_phi),
        int(num_species),
        int(num_triplets),
        float(zero_tol),
        thread_g3,
        thread_g2,
    )
    # Reduce per-thread buffers outside numba (numpy's sum is much
    # faster than an explicit 5D numba loop).
    g3count = thread_g3.sum(axis=0)
    g2count = thread_g2.sum(axis=0)
    return g3count, g2count
