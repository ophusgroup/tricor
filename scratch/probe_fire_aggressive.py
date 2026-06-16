"""Probe whether more-aggressive FIRE settings give larger per-step |Δr|
without losing fmax descent or causing wall blow-ups.

Runs a tiny sweep over (dt, maxstep) on ONE amorphous SiO2 trajectory.
Amorphous because it had the worst fmax growth in the v3_fire sweep with
maxstep=0.3 + steps=150 — so it's the most sensitive canary for any
setting that introduces wall oscillation.

For each config, runs 80 FIRE steps with MACE+wall and records per-step:
  - mean |Δr| (the training-target magnitude)
  - max  |Δr| (catch overshoots that hit maxstep cap)
  - energy and fmax

Outputs a single per-config row plus a CSV with per-step traces, so we
can pick the production setting from data rather than vibes.

Run (on a free GPU — pick one not running training):
    CUDA_VISIBLE_DEVICES=0 \\
      /home/ehrdt/miniforge3/envs/mace/bin/python scratch/probe_fire_aggressive.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from ase.optimize import FIRE

import tricor as tc
from mace.calculators import mace_mp

sys.path.insert(0, "/home/ehrdt/tricor/mace")
from wall_calculator import MinDistanceWallCalculator, per_pair_min_from_atoms  # noqa: E402


# === CONFIG ============================================================
# Reference: same setup as the production pilot at amorphous SiO2.
REFERENCE_CIF = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos/mp-7000_SiO2.cif")
CELL_DIM = 50.0
RELATIVE_DENSITY = 0.92
REGIME = "amorphous"
RNG_SEED = 42
N_STEPS = 80
WALL_K = 1000.0
WALL_EXPONENT = 4
WALL_MARGIN = 0.0

OUT_DIR = Path("/home/ehrdt/tricor/scratch")
TRACE_CSV = OUT_DIR / "probe_fire_aggressive_traces.csv"
SUMMARY_CSV = OUT_DIR / "probe_fire_aggressive_summary.csv"

# (label, FIRE kwargs).  None for maxstep means uncapped.
CONFIGS = [
    ("baseline_prod",     dict(dt=0.1, maxstep=0.3)),                   # current production
    ("dt0.3_step0.3",     dict(dt=0.3, maxstep=0.3)),                   # 3x dt, same cap
    ("dt0.3_step0.5",     dict(dt=0.3, maxstep=0.5)),                   # 3x dt, bigger cap
    ("dt0.5_step0.5",     dict(dt=0.5, maxstep=0.5)),                   # 5x dt, bigger cap
    ("dt0.1_uncapped",    dict(dt=0.1, maxstep=None)),                  # baseline dt, no cap
    ("dt0.3_uncapped",    dict(dt=0.3, maxstep=None)),                  # 3x dt, no cap
    ("fast_ramp_Nmin2",   dict(dt=0.1, maxstep=0.3, Nmin=2)),           # quicker dt ramp
]
# =======================================================================


def _min_image(delta, cell):
    inv = np.linalg.inv(cell)
    df = delta @ inv.T
    df -= np.round(df)
    return df @ cell


def build_initial_state():
    """Same packing + bond_relax cleanup as the production generator."""
    ref = read(REFERENCE_CIF)
    shell = tc.CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)
    sc = tc.Supercell.from_atoms(
        ref, cell_dim_angstroms=CELL_DIM,
        r_max=10, r_step=0.1, phi_num_bins=90,
        relative_density=RELATIVE_DENSITY, rng_seed=RNG_SEED,
        label=f"probe_{REGIME}",
    )
    preset = dict(tc.Supercell.PRESETS[REGIME])
    preset["displacement_sigma"] = 0.0
    sc.generate(shell, **{**preset, "num_steps": 0},
                refine_orientations=False, show_progress=False)
    sc.bond_relax(shell, n_iter=80, max_step=0.1)
    return sc.atoms


def run_one(label, fire_kwargs, base_calc, atoms_template, trace_writer):
    """Run N_STEPS of MACE+wall+FIRE with the given config; return summary dict."""
    # Fresh copy of the starting atoms (deepcopy via Atoms constructor).
    from ase import Atoms
    atoms = Atoms(
        numbers=atoms_template.numbers,
        positions=atoms_template.positions.copy(),
        cell=atoms_template.cell.array.copy(),
        pbc=atoms_template.pbc,
    )

    r_min = per_pair_min_from_atoms(atoms, margin=WALL_MARGIN)
    atoms.calc = MinDistanceWallCalculator(
        base_calc=base_calc, r_min_per_pair=r_min,
        k=WALL_K, exponent=WALL_EXPONENT,
    )

    # Initial state
    e0 = float(atoms.get_potential_energy())
    f0 = float(np.abs(atoms.get_forces()).max())
    prev_pos = atoms.positions.copy()

    # Drop FIRE kwargs into ASE's optimizer; only pass non-None.
    kwargs = {k: v for k, v in fire_kwargs.items() if v is not None}
    opt = FIRE(atoms, logfile=None, **kwargs)

    per_step_dr_mean: list[float] = []
    per_step_dr_max: list[float] = []
    per_step_e: list[float] = [e0]
    per_step_f: list[float] = [f0]

    def cb():
        nonlocal prev_pos
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        dxs = _min_image(atoms.positions - prev_pos, cell)
        dr = np.linalg.norm(dxs, axis=-1)
        per_step_dr_mean.append(float(dr.mean()))
        per_step_dr_max.append(float(dr.max()))
        per_step_e.append(float(atoms.get_potential_energy()))
        per_step_f.append(float(np.abs(atoms.get_forces()).max()))
        prev_pos = atoms.positions.copy()

    opt.attach(cb, interval=1)
    t0 = time.perf_counter()
    opt.run(fmax=0.01, steps=N_STEPS)
    elapsed = time.perf_counter() - t0

    # Per-step traces → CSV
    for step in range(len(per_step_dr_mean)):
        trace_writer.writerow({
            "config": label,
            "step": step + 1,
            "dr_mean": per_step_dr_mean[step],
            "dr_max":  per_step_dr_max[step],
            "energy": per_step_e[step + 1],
            "fmax":   per_step_f[step + 1],
        })

    summary = {
        "config": label,
        "n_steps_taken": int(opt.nsteps),
        "n_atoms": int(len(atoms)),
        "elapsed_s": round(elapsed, 1),
        "e_start": round(e0, 2),
        "e_end":   round(per_step_e[-1], 2),
        "delta_e": round(per_step_e[-1] - e0, 2),
        "fmax_start": round(f0, 3),
        "fmax_end":   round(per_step_f[-1], 3),
        "dr_mean_avg": round(float(np.mean(per_step_dr_mean)), 5),
        "dr_mean_early": round(float(np.mean(per_step_dr_mean[:10])), 5),
        "dr_mean_late":  round(float(np.mean(per_step_dr_mean[-10:])), 5),
        "dr_max_overall": round(float(np.max(per_step_dr_max)), 4),
    }
    return summary


def main():
    print(f"Probe — FIRE aggressive-settings sweep on {REGIME} SiO2 at CELL={CELL_DIM}")
    print(f"Configs: {[c[0] for c in CONFIGS]}")

    print("\nBuilding initial structure (one-shot, reused across configs)...")
    atoms_template = build_initial_state()
    print(f"  {len(atoms_template)} atoms")

    print("\nInitializing MACE-MPA medium (cold start ~30s)...")
    base_calc = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float32")
    print("MACE ready.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    with open(TRACE_CSV, "w", newline="") as tf:
        trace_writer = csv.DictWriter(
            tf,
            fieldnames=["config", "step", "dr_mean", "dr_max", "energy", "fmax"],
            lineterminator="\n",
        )
        trace_writer.writeheader()
        for label, kwargs in CONFIGS:
            print(f"\n=== {label}  ({kwargs}) ===")
            try:
                summary = run_one(label, kwargs, base_calc, atoms_template, trace_writer)
                summaries.append(summary)
                print(
                    f"  steps={summary['n_steps_taken']}  "
                    f"|Δr| mean avg={summary['dr_mean_avg']}  "
                    f"early={summary['dr_mean_early']}  late={summary['dr_mean_late']}  "
                    f"max={summary['dr_max_overall']}  "
                    f"E {summary['e_start']}→{summary['e_end']} (Δ {summary['delta_e']})  "
                    f"fmax {summary['fmax_start']}→{summary['fmax_end']}  "
                    f"{summary['elapsed_s']}s"
                )
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(SUMMARY_CSV, "w", newline="") as f:
        if summaries:
            w = csv.DictWriter(
                f, fieldnames=list(summaries[0].keys()), lineterminator="\n",
            )
            w.writeheader()
            w.writerows(summaries)
    print(f"\nWrote summary: {SUMMARY_CSV}")
    print(f"Wrote traces:  {TRACE_CSV}")


if __name__ == "__main__":
    main()
