"""Parity + benchmark for vectorized vs loop MinDistanceWallCalculator.

Loads a frame from an existing pilot NPZ, reconstructs the atoms object,
computes per-pair r_min thresholds with per_pair_min_from_atoms, then runs
both calculator implementations side-by-side.

Verifies:
  - Energy parity (atol)
  - Force parity (atol/rtol)
  - Wall metadata parity (E_wall, n_violations, max_penetration)
  - Speedup factor over N timed calls

Uses a ZeroCalc as the base calculator so the bench measures wall cost only
(strips MACE GPU forward-pass time from the comparison — that's the part
that doesn't change).

Run with: /home/ehrdt/miniforge3/envs/mace/bin/python scratch/bench_wall_calc.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

# Make mace/wall_calculator.py importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mace"))
from wall_calculator import (  # noqa: E402
    MinDistanceWallCalculator,
    MinDistanceWallCalculatorLoop,
    per_pair_min_from_atoms,
)


# === CONFIG ============================================================
# A SiC SRO trajectory from the running pilot (~11k atoms, ~tens of
# thousands of in-cutoff pairs — representative of the dominant cost).
NPZ_PATH = "/home/ehrdt/tricor/mace/data/pilot_v1/train/SiC/SiC_SRO_cell050_idx00026_seed001100026.npz"
# Frames to test: 0 = initial post-bond_relax (many violations), middle and
# late-step frames should have progressively fewer.
FRAME_INDICES = [0, 5, 10, -1]
N_TIMING_CALLS = 20         # ~1 trajectory's worth of LBFGS steps
WALL_MARGIN = 0.0
WALL_K = 1000.0
WALL_EXPONENT = 4
PARITY_ATOL = 1e-9          # absolute tolerance for energy + forces
PARITY_RTOL = 1e-10
# =======================================================================


class ZeroCalc(Calculator):
    """Returns E=0, F=0 — strips base-calc cost so the bench is wall-only."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros_like(atoms.positions, dtype=np.float64)


def load_frame(npz_path: str, frame_index: int) -> Atoms:
    d = np.load(npz_path)
    positions = d["positions"][frame_index]
    cell = d["cell"]
    numbers = d["species_numbers"]
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def run_one(calc, atoms):
    calc.calculate(atoms, ("energy", "forces"), all_changes)
    return (
        float(calc.results["energy"]),
        np.asarray(calc.results["forces"]).copy(),
        float(calc.results["wall_energy"]),
        int(calc.results["wall_n_violations"]),
        float(calc.results["wall_max_penetration"]),
    )


def time_calc(calc, atoms, n_calls):
    # Warm up once (allocation, caching).
    calc.calculate(atoms, ("energy", "forces"), all_changes)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        calc.calculate(atoms, ("energy", "forces"), all_changes)
    return (time.perf_counter() - t0) / n_calls


def main():
    print(f"NPZ: {NPZ_PATH}")
    print(f"Frames: {FRAME_INDICES}\n")

    # Build r_min once from frame 0 (matches generator behavior).
    atoms0 = load_frame(NPZ_PATH, 0)
    r_min = per_pair_min_from_atoms(atoms0, margin=WALL_MARGIN)
    print(f"r_min ({len(r_min)} entries): "
          f"{ {k: round(v, 4) for k, v in r_min.items()} }")
    print(f"N atoms = {len(atoms0)}\n")

    base_loop = ZeroCalc()
    base_vec = ZeroCalc()
    calc_loop = MinDistanceWallCalculatorLoop(
        base_loop, r_min, k=WALL_K, exponent=WALL_EXPONENT,
    )
    calc_vec = MinDistanceWallCalculator(
        base_vec, r_min, k=WALL_K, exponent=WALL_EXPONENT,
    )

    all_passed = True
    timings = []  # (frame, n_viol, t_loop, t_vec)

    for frame in FRAME_INDICES:
        atoms = load_frame(NPZ_PATH, frame)
        E_l, F_l, Ew_l, nv_l, mp_l = run_one(calc_loop, atoms)
        E_v, F_v, Ew_v, nv_v, mp_v = run_one(calc_vec, atoms)

        dE = abs(E_l - E_v)
        dF_max = float(np.max(np.abs(F_l - F_v)))
        d_Ew = abs(Ew_l - Ew_v)
        d_mp = abs(mp_l - mp_v)

        ok_E = dE < PARITY_ATOL
        ok_F = np.allclose(F_l, F_v, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        ok_Ew = d_Ew < PARITY_ATOL
        ok_nv = nv_l == nv_v
        ok_mp = d_mp < 1e-12
        ok = ok_E and ok_F and ok_Ew and ok_nv and ok_mp
        all_passed = all_passed and ok

        print(f"--- frame {frame} ---")
        print(f"  n_violations  loop={nv_l:>7d}  vec={nv_v:>7d}   match={ok_nv}")
        print(f"  E_wall        loop={Ew_l:.6e}  vec={Ew_v:.6e}   |d|={d_Ew:.3g}  ok={ok_Ew}")
        print(f"  E_total       loop={E_l:.6e}  vec={E_v:.6e}   |d|={dE:.3g}  ok={ok_E}")
        print(f"  |F_l-F_v|max  = {dF_max:.3g}   ok={ok_F}")
        print(f"  max_penetr    loop={mp_l:.6e}  vec={mp_v:.6e}   ok={ok_mp}")
        print(f"  PARITY: {'OK' if ok else 'FAIL'}")

        t_loop = time_calc(calc_loop, atoms, N_TIMING_CALLS)
        t_vec = time_calc(calc_vec, atoms, N_TIMING_CALLS)
        timings.append((frame, nv_v, t_loop, t_vec))
        print(f"  per-call timings ({N_TIMING_CALLS} calls each)")
        print(f"    loop : {t_loop * 1000:9.3f} ms")
        print(f"    vec  : {t_vec * 1000:9.3f} ms")
        print(f"    speedup: {t_loop / max(t_vec, 1e-12):.1f}x\n")

    print("=" * 60)
    print(f"OVERALL PARITY: {'OK' if all_passed else 'FAIL'}\n")
    print(f"{'frame':>6}  {'n_viol':>8}  {'loop (ms)':>10}  {'vec (ms)':>10}  {'speedup':>8}")
    for frame, nv, tl, tv in timings:
        print(f"{frame:>6}  {nv:>8d}  {tl * 1000:>10.3f}  {tv * 1000:>10.3f}"
              f"  {tl / max(tv, 1e-12):>7.1f}x")
    # Sum the loop cost across the trajectory steps as a proxy for total
    # wall time saved per trajectory.
    avg_loop = sum(t for _, _, t, _ in timings) / len(timings)
    avg_vec = sum(t for _, _, _, t in timings) / len(timings)
    per_traj_loop = avg_loop * N_TIMING_CALLS
    per_traj_vec = avg_vec * N_TIMING_CALLS
    print(f"\nEstimated per-trajectory wall cost ({N_TIMING_CALLS} LBFGS steps,"
          f" avg of tested frames):")
    print(f"  loop : {per_traj_loop:6.2f} s")
    print(f"  vec  : {per_traj_vec:6.2f} s")
    print(f"  saved: {per_traj_loop - per_traj_vec:6.2f} s/traj")


if __name__ == "__main__":
    main()
