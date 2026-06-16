"""Subprofile the wall calculator hot path to find where the ~180 ms/call
actually goes. The parity bench showed vectorization only buys 1.1x —
meaning the per-pair Python loop isn't the bottleneck. Suspects:
neighbor_list build, ASE Calculator overhead (atoms.copy), or np.add.at.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mace"))
from wall_calculator import per_pair_min_from_atoms  # noqa: E402


# === CONFIG ============================================================
NPZ_PATH = "/home/ehrdt/tricor/mace/data/pilot_v1/train/SiC/SiC_SRO_cell050_idx00026_seed001100026.npz"
FRAME_INDEX = -1   # final frame — most violations
N_CALLS = 20
WALL_K = 1000.0
WALL_EXPONENT = 4
# =======================================================================


def load_frame(npz_path: str, frame_index: int) -> Atoms:
    d = np.load(npz_path)
    return Atoms(
        numbers=d["species_numbers"],
        positions=d["positions"][frame_index],
        cell=d["cell"],
        pbc=True,
    )


def avg_time(fn, n=N_CALLS):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0  # ms


def main():
    atoms = load_frame(NPZ_PATH, FRAME_INDEX)
    r_min = per_pair_min_from_atoms(atoms, margin=0.0)
    cutoff = max(r_min.values())
    print(f"N atoms = {len(atoms)}")
    print(f"cutoff = {cutoff:.3f} Å")
    print(f"frame index = {FRAME_INDEX}\n")

    # Build the r_min table once so we can time just the lookup.
    zmax = max(max(k) for k in r_min)
    table = np.zeros((zmax + 1, zmax + 1), dtype=np.float64)
    for (a, b), v in r_min.items():
        table[a, b] = v
        table[b, a] = v
    z = atoms.numbers

    # 1. Neighbor list with the four-array signature actually used.
    def t_neighbor_list_ijDd():
        neighbor_list("ijDd", atoms, cutoff, self_interaction=False)
    t1 = avg_time(t_neighbor_list_ijDd)

    # 2. Neighbor list returning just distance (no D_vec) — does the
    # D_vec cost extra?
    def t_neighbor_list_ijd():
        neighbor_list("ijd", atoms, cutoff, self_interaction=False)
    t2 = avg_time(t_neighbor_list_ijd)

    # 3. ASE atoms.copy() — Calculator.calculate triggers two of these.
    def t_atoms_copy():
        atoms.copy()
    t3 = avg_time(t_atoms_copy)

    # 4. Vectorized per-pair work alone (pre-built neighbor list).
    i_idx, j_idx, D_vec, dist = neighbor_list(
        "ijDd", atoms, cutoff, self_interaction=False,
    )
    n_exp = WALL_EXPONENT
    kw = WALL_K
    N = len(atoms)

    def t_vectorized_pairwise():
        F = np.zeros((N, 3), dtype=np.float64)
        zi = z[i_idx]
        zj = z[j_idx]
        r_min_arr = table[zi, zj]
        pen = r_min_arr - dist
        viol = pen > 0
        if viol.any():
            pen_v = pen[viol]
            i_v = i_idx[viol]
            D_v = D_vec[viol]
            dist_v = dist[viol]
            V_arr = kw * pen_v ** n_exp
            f_mag = kw * n_exp * (pen_v ** (n_exp - 1))
            inv_r = 1.0 / np.maximum(dist_v, 1e-12)
            force = -(f_mag * inv_r)[:, None] * D_v
            np.add.at(F, i_v, force)
            return float(V_arr.sum()) * 0.5
        return 0.0
    t4 = avg_time(t_vectorized_pairwise)

    # 5. Just np.add.at on the violators.
    zi = z[i_idx]
    zj = z[j_idx]
    r_min_arr = table[zi, zj]
    pen = r_min_arr - dist
    viol = pen > 0
    i_v = i_idx[viol]
    D_v = D_vec[viol]
    dist_v = dist[viol]
    pen_v = pen[viol]
    f_mag = kw * n_exp * (pen_v ** (n_exp - 1))
    inv_r = 1.0 / np.maximum(dist_v, 1e-12)
    force = -(f_mag * inv_r)[:, None] * D_v
    print(f"\n[violation stats] n_violations={len(i_v)}  "
          f"i_v unique={len(set(i_v.tolist()))}")

    def t_scatter_add_at():
        F = np.zeros((N, 3), dtype=np.float64)
        np.add.at(F, i_v, force)
    t5 = avg_time(t_scatter_add_at)

    # 6. Same scatter via bincount (per-component, 3 calls).
    def t_scatter_bincount():
        F = np.zeros((N, 3), dtype=np.float64)
        F[:, 0] = np.bincount(i_v, weights=force[:, 0], minlength=N)
        F[:, 1] = np.bincount(i_v, weights=force[:, 1], minlength=N)
        F[:, 2] = np.bincount(i_v, weights=force[:, 2], minlength=N)
    t6 = avg_time(t_scatter_bincount)

    print()
    print(f"{'task':45s} {'avg ms':>10s}")
    print("-" * 60)
    print(f"{'neighbor_list ijDd (i,j,D_vec,dist)':45s} {t1:>10.3f}")
    print(f"{'neighbor_list ijd  (i,j,dist only)':45s} {t2:>10.3f}")
    print(f"{'atoms.copy()':45s} {t3:>10.3f}")
    print(f"{'vectorized pairwise (pre-built nlist)':45s} {t4:>10.3f}")
    print(f"{'  ↳ np.add.at scatter only':45s} {t5:>10.3f}")
    print(f"{'  ↳ np.bincount scatter (3 calls)':45s} {t6:>10.3f}")


if __name__ == "__main__":
    main()
