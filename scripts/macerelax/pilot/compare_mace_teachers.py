"""Pilot — compare MACE-MPA vs MACE-OMAT as teachers on U-affected metals.

Research finding (project_mace_relax_ddp.md → MACE_RELAX_DDP.md): MACE-MPA
exhibits "spurious underbinding for every Hubbard-U-corrected metal" because
Materials Project applies +U only to oxides/fluorides of {V, Cr, Mn, Fe, Co,
Ni, Mo, W}.  The MLST paper (DOI 10.1088/2632-2153/ae6be5) reports MAE ~3.8
eV for MACE-MP-0b3 on U-corrected metals vs 0.68 eV for the no-U MatPES
model.  MACE-OMAT-0 was trained on OMat24 which doesn't carry MP's
inconsistent +U scheme — it should be free of this pathology.

This pilot tests, on 7 representative materials, whether MACE-OMAT-0
produces visibly better FIRE relaxation trajectories than MACE-MPA-0 on
U-affected oxides.  If yes, we can include these chemistries in the
training corpus by switching teacher (or generating each material twice).
If the difference is marginal, keep them excluded.

Test set
--------
U-affected oxides (5):
  - mp-1181546_Fe3O4   magnetite, mixed Fe²⁺/³⁺
  - mp-19079_CoO       rocksalt cobalt oxide
  - mp-18759_Mn3O4     hausmannite, Jahn-Teller Mn³⁺
  - mp-20593_VO2       vanadium dioxide, Mott–Peierls
  - mp-754806_Ni5O6    nickel oxide, tests Ni
Controls (2):
  - mp-2657_TiO2       rutile — Ti not U-corrected, same teacher behavior expected
  - mp-1143_Al2O3      corundum — non-TM, same teacher behavior expected

7 materials × 2 teachers × 2 seeds × 1 regime (amorphous) = 28 trajectories
At ~35 Å cells and N_STEPS=40, each trajectory is ~2 min → ~1 GPU-hr total.

Outputs
-------
OUTPUT_ROOT/
  ├── mpa/<system>_<seed>.npz       per-trajectory: positions, loss, fmax, ...
  ├── omat/<system>_<seed>.npz      same for OMAT teacher
  └── summary.csv                    one row per (teacher, system, seed) with
                                     key metrics:
                                       converged, final_loss, final_fmax,
                                       n_steps, mean_metal_oxygen_bond,
                                       min_atom_distance_final, runtime_sec

Analysis script (separate): scripts/macerelax/pilot/analyze_teachers.py.

Notes / safety
--------------
* The OMAT model name "medium-omat-0" matches the mace-foundations naming
  convention.  If your installed mace version differs, override
  TEACHERS["omat"]["model_name"] below.  Cold-start download is ~250 MB.
* Won't pollute the production dataset — OUTPUT_ROOT is a separate dir.
* If a CIF is missing or a teacher fails to load, that combination is
  logged and skipped; other combinations continue.

Run with:
    /home/ehrdt/miniforge3/envs/mace/bin/python \\
        scripts/macerelax/pilot/compare_mace_teachers.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# --- I/O ---
CIF_DIR     = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV_training")
OUTPUT_ROOT = Path("/home/ehrdt/tricor/mace/data/pilot_teacher_comparison")

# --- Test materials (filenames in CIF_DIR) ---
# Each entry is (system_label, cif_filename, is_u_affected_oxide).
# system_label is used for output filenames; should be filesystem-safe.
TEST_MATERIALS = [
    # U-affected oxides — the main test
    ("Fe3O4",  "mp-1181546_Fe3O4.cif",  True),
    ("CoO",    "mp-19079_CoO.cif",      True),
    ("Mn3O4",  "mp-18759_Mn3O4.cif",    True),
    ("VO2",    "mp-20593_VO2.cif",      True),
    ("Ni5O6",  "mp-754806_Ni5O6.cif",   True),
    # Controls — teachers should behave the same here
    ("TiO2",   "mp-2657_TiO2.cif",      False),  # non-U TM oxide
    ("Al2O3",  "mp-1143_Al2O3.cif",     False),  # non-TM oxide
]

# --- Teachers ---
TEACHERS = {
    "mpa":  {"model_name": "medium-mpa-0",  "dtype": "float32"},
    "omat": {"model_name": "medium-omat-0", "dtype": "float32"},
}

# --- Generation parameters (smaller than production for speed) ---
REGIME       = "amorphous"
SEEDS        = [42, 43]              # 2 seeds per (material, teacher)
CELL_SIZE    = 35.0                  # Å — was 50.0 in production
RELATIVE_DENSITY = 0.92              # standard amorphous density
N_STEPS      = 40                    # was 60 in production
OPT_MAXSTEP  = 0.3
FMAX_TARGET  = 0.05

# --- Cleanup ---
BOND_RELAX_N_ITER   = 80
BOND_RELAX_MAX_STEP = 0.1

# --- Wall ---
WALL_K        = 1000.0
WALL_EXPONENT = 4
WALL_MARGIN   = 0.0

# --- Device ---
MACE_DEVICE = "cuda:1"

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# Pre-import torch so we can fail fast if CUDA isn't available.
import torch
if MACE_DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit(f"[abort] MACE_DEVICE={MACE_DEVICE!r} but torch.cuda.is_available() is False")
if MACE_DEVICE.startswith("cuda:"):
    _idx = int(MACE_DEVICE.split(":")[1])
    if _idx >= torch.cuda.device_count():
        raise SystemExit(
            f"[abort] MACE_DEVICE={MACE_DEVICE!r} but only {torch.cuda.device_count()} "
            f"GPU(s) visible"
        )

from ase.io import read as ase_read
from ase.optimize import FIRE

import tricor as tc
from mace.calculators import mace_mp

# The wall calculator lives in the generation/ dir — add it to sys.path.
_GEN_DIR = Path(__file__).resolve().parent.parent / "generation"
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))
from wall_calculator import MinDistanceWallCalculator, per_pair_min_from_atoms


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers (lifted from generate_mace_trajectories.py, simplified)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """Per-(teacher, system, seed) result row."""
    teacher:        str
    system:         str
    seed:           int
    cif_path:       str
    is_u_affected:  bool
    n_atoms:        int
    converged:      bool
    n_steps:        int
    initial_loss:   float
    final_loss:     float
    best_loss:      float
    fmax_initial:   float
    fmax_final:     float
    min_d_initial:  float
    min_d_final:    float
    mean_M_O_bond:  Optional[float]   # mean transition-metal–O bond length, signature of U pathology
    runtime_sec:    float
    error:          str = ""


def build_supercell(cif_path: Path, regime: str, rho: float, seed: int):
    """Pack a tricor supercell at the requested regime/density."""
    ref = ase_read(str(cif_path), format="cif")
    shell = tc.CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)

    # Build the per-CIF G3 distribution (cached structure descriptor).
    from tricor import G3Distribution
    dist = G3Distribution(ref, label=str(cif_path.stem))
    dist.measure_g3(r_max=10.0, r_step=0.1, phi_num_bins=90, show_progress=False)

    cell = tc.Supercell(
        dist,
        cell_dim_angstroms=(CELL_SIZE,) * 3,
        relative_density=rho,
        rng_seed=seed,
        label=f"{cif_path.stem}_{regime}_{seed}",
    )
    # Use tricor's amorphous preset (displacement_sigma=0 to avoid noise).
    preset = dict(tc.Supercell.PRESETS[regime])
    preset["displacement_sigma"] = 0.0
    preset["num_steps"] = 0   # skip shell_relax — bond_relax cleanup follows
    cell.generate(shell, **preset, refine_orientations=False, show_progress=False)
    return cell, shell, ref


def cleanup(cell, shell):
    """bond_relax cleanup to remove overlaps before MACE sees the cell."""
    cell.bond_relax(shell,
                    n_iter=BOND_RELAX_N_ITER,
                    max_step=BOND_RELAX_MAX_STEP)


def run_mace_fire(atoms, base_calc):
    """FIRE on MACE+wall PES.  Returns dict with positions / loss / fmax."""
    r_min_per_pair = per_pair_min_from_atoms(atoms, margin=WALL_MARGIN)
    atoms.calc = MinDistanceWallCalculator(
        base_calc=base_calc, r_min_per_pair=r_min_per_pair,
        k=WALL_K, exponent=WALL_EXPONENT,
    )

    snapshots:  list[np.ndarray] = []
    loss_hist:  list[float]      = []
    fmax_hist:  list[float]      = []
    best = {"E": float("inf"), "pos": atoms.positions.copy()}

    e0 = float(atoms.get_potential_energy())
    f0 = atoms.get_forces()
    fmax0 = float(np.abs(f0).max())

    snapshots.append(atoms.positions.copy().astype(np.float32))
    loss_hist.append(e0)
    fmax_hist.append(fmax0)
    best["E"] = e0
    best["pos"] = atoms.positions.copy()

    opt = FIRE(atoms, maxstep=OPT_MAXSTEP, logfile=None)

    def per_step_cb():
        try:
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
            fmax = float(np.abs(f).max())
        except Exception:
            e   = float("nan")
            fmax = float("nan")
        loss_hist.append(e)
        fmax_hist.append(fmax)
        if e < best["E"]:
            best["E"] = e
            best["pos"] = atoms.positions.copy()
        snapshots.append(atoms.positions.copy().astype(np.float32))

    opt.attach(per_step_cb, interval=1)
    opt.run(fmax=FMAX_TARGET, steps=N_STEPS)

    return {
        "positions":      np.stack(snapshots, axis=0),
        "loss":           np.asarray(loss_hist, dtype=np.float64),
        "fmax":           np.asarray(fmax_hist, dtype=np.float64),
        "best_positions": best["pos"].astype(np.float32),
        "n_steps_run":    int(opt.nsteps),
        "fmax_initial":   fmax0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics — Hubbard-U pathology signature
# ─────────────────────────────────────────────────────────────────────────────

U_METALS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Mo", "W"}


def mean_metal_oxygen_bond(atoms, max_r: float = 3.0) -> Optional[float]:
    """Mean nearest-neighbor distance between any U-affected metal atom and
    any oxygen, in Angstroms.  This is the canonical signature of the
    Hubbard-U pathology: if MACE-MPA over-repels U-corrected M–O pairs, the
    mean M–O bond will be longer than the equilibrium value.

    Returns None if no (M, O) atom pair exists in the cell.
    """
    syms = np.array(atoms.get_chemical_symbols())
    pos  = atoms.get_positions()
    cell = atoms.get_cell()

    metal_mask = np.isin(syms, list(U_METALS))
    oxygen_mask = syms == "O"
    if not metal_mask.any() or not oxygen_mask.any():
        return None

    metal_idx = np.where(metal_mask)[0]
    oxygen_idx = np.where(oxygen_mask)[0]

    # Min-image distance from each metal atom to its nearest oxygen.
    inv_cell = np.linalg.inv(cell)
    bonds = []
    for mi in metal_idx:
        deltas = pos[oxygen_idx] - pos[mi]
        frac = deltas @ inv_cell
        frac = frac - np.round(frac)
        deltas = frac @ cell
        dists = np.linalg.norm(deltas, axis=1)
        nearest = dists.min()
        if nearest <= max_r:
            bonds.append(nearest)
    if not bonds:
        return None
    return float(np.mean(bonds))


def min_atom_distance(atoms) -> float:
    """Min interatomic distance under PBC, in Angstroms.  Used to detect
    close-contact-trap fallout (MACE-MP-0 paper, §3.3)."""
    pos  = atoms.get_positions()
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    n = len(pos)
    if n < 2:
        return float("inf")
    # Pairwise min-image distances — O(n²), fine for ≤3000-atom cells.
    deltas = pos[:, None, :] - pos[None, :, :]
    frac = deltas @ inv_cell
    frac = frac - np.round(frac)
    deltas = frac @ cell
    dists = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return float(dists.min())


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_one(system_label: str, cif_path: Path, is_u_affected: bool,
            teacher_label: str, calc, seed: int) -> Trajectory:
    """Build cell, cleanup, MACE+FIRE relax, compute metrics.

    Always returns a Trajectory; on error the .error field is set and other
    fields are NaN.  Never raises (errors are caught + logged).
    """
    t0 = time.time()
    try:
        # 1. Build packed supercell with tricor.
        cell, shell, ref = build_supercell(
            cif_path, REGIME, RELATIVE_DENSITY, seed,
        )
        cleanup(cell, shell)
        atoms = cell.atoms
        n_atoms = len(atoms)
        min_d_initial = min_atom_distance(atoms)

        # 2. MACE+wall FIRE.
        result = run_mace_fire(atoms, calc)

        # 3. Per-step metrics + final diagnostics.
        atoms.set_positions(result["best_positions"])
        min_d_final = min_atom_distance(atoms)
        mean_MO = mean_metal_oxygen_bond(atoms) if is_u_affected else None

        loss     = result["loss"]
        fmax     = result["fmax"]
        converged = bool(fmax[-1] < FMAX_TARGET)

        return Trajectory(
            teacher=teacher_label,
            system=system_label,
            seed=seed,
            cif_path=str(cif_path),
            is_u_affected=is_u_affected,
            n_atoms=n_atoms,
            converged=converged,
            n_steps=result["n_steps_run"],
            initial_loss=float(loss[0]),
            final_loss=float(loss[-1]),
            best_loss=float(loss.min()),
            fmax_initial=result["fmax_initial"],
            fmax_final=float(fmax[-1]),
            min_d_initial=min_d_initial,
            min_d_final=min_d_final,
            mean_M_O_bond=mean_MO,
            runtime_sec=time.time() - t0,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(limit=3, file=sys.stdout)
        return Trajectory(
            teacher=teacher_label, system=system_label, seed=seed,
            cif_path=str(cif_path), is_u_affected=is_u_affected,
            n_atoms=-1, converged=False, n_steps=0,
            initial_loss=float("nan"), final_loss=float("nan"),
            best_loss=float("nan"),
            fmax_initial=float("nan"), fmax_final=float("nan"),
            min_d_initial=float("nan"), min_d_final=float("nan"),
            mean_M_O_bond=None,
            runtime_sec=time.time() - t0,
            error=err,
        )


def save_trajectory_npz(out_dir: Path, traj: Trajectory, result: dict,
                         atoms) -> Path:
    """Persist per-step positions + loss + metadata for a trajectory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{traj.system}_seed{traj.seed}.npz"
    path = out_dir / fname
    np.savez_compressed(
        path,
        positions=result["positions"],
        loss=result["loss"],
        fmax=result["fmax"],
        species=np.array(atoms.get_chemical_symbols()),
        cell=atoms.get_cell().array,
        meta=json.dumps(asdict(traj)),
    )
    return path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Pre-flight checks ─────────────────────────────────────────────────
    missing = []
    for label, fname, _ in TEST_MATERIALS:
        if not (CIF_DIR / fname).is_file():
            missing.append(fname)
    if missing:
        print(f"[abort] missing CIF files in {CIF_DIR}:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)
    print(f"[pilot] CIF_DIR     : {CIF_DIR}")
    print(f"[pilot] OUTPUT_ROOT : {OUTPUT_ROOT}")
    print(f"[pilot] {len(TEST_MATERIALS)} materials × {len(TEACHERS)} teachers "
          f"× {len(SEEDS)} seeds = "
          f"{len(TEST_MATERIALS) * len(TEACHERS) * len(SEEDS)} trajectories")

    summary_rows: list[dict] = []

    # ── For each teacher: load once, then iterate materials × seeds ──────
    for teacher_label, teacher_cfg in TEACHERS.items():
        out_dir = OUTPUT_ROOT / teacher_label
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Teacher: {teacher_label}  "
              f"(model={teacher_cfg['model_name']}) ===")
        print(f"Loading MACE (cold start ~30s if first time)...")
        try:
            calc = mace_mp(
                model=teacher_cfg["model_name"],
                device=MACE_DEVICE,
                default_dtype=teacher_cfg["dtype"],
            )
            print(f"MACE ready.")
        except Exception as exc:
            print(f"[skip-teacher] failed to load {teacher_label}: {exc}")
            traceback.print_exc(limit=3, file=sys.stdout)
            continue

        for (sys_label, cif_fname, is_u) in TEST_MATERIALS:
            cif_path = CIF_DIR / cif_fname
            for seed in SEEDS:
                print(f"  [{teacher_label}/{sys_label}/seed{seed}] running...",
                      flush=True)
                # Rerun the cell-build + relax + save sequence.
                # We save the trajectory data inside run_one's exception
                # handler is awkward; instead, run + return Trajectory, and
                # if non-error, redo the (cheap) atoms snapshot from cell.
                t0 = time.time()
                try:
                    cell, shell, _ref = build_supercell(
                        cif_path, REGIME, RELATIVE_DENSITY, seed,
                    )
                    cleanup(cell, shell)
                    atoms = cell.atoms
                    n_atoms = len(atoms)
                    min_d_initial = min_atom_distance(atoms)

                    result = run_mace_fire(atoms, calc)

                    atoms.set_positions(result["best_positions"])
                    min_d_final = min_atom_distance(atoms)
                    mean_MO = mean_metal_oxygen_bond(atoms) if is_u else None
                    fmax = result["fmax"]
                    loss = result["loss"]

                    traj = Trajectory(
                        teacher=teacher_label,
                        system=sys_label,
                        seed=seed,
                        cif_path=str(cif_path),
                        is_u_affected=is_u,
                        n_atoms=n_atoms,
                        converged=bool(fmax[-1] < FMAX_TARGET),
                        n_steps=result["n_steps_run"],
                        initial_loss=float(loss[0]),
                        final_loss=float(loss[-1]),
                        best_loss=float(loss.min()),
                        fmax_initial=result["fmax_initial"],
                        fmax_final=float(fmax[-1]),
                        min_d_initial=min_d_initial,
                        min_d_final=min_d_final,
                        mean_M_O_bond=mean_MO,
                        runtime_sec=time.time() - t0,
                    )
                    save_trajectory_npz(out_dir, traj, result, atoms)
                    print(f"    ✓ converged={traj.converged} "
                          f"final_loss={traj.final_loss:.3e} "
                          f"mean_M-O={mean_MO if mean_MO else 'n/a'} "
                          f"dt={traj.runtime_sec:.1f}s")
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc(limit=3, file=sys.stdout)
                    traj = Trajectory(
                        teacher=teacher_label, system=sys_label, seed=seed,
                        cif_path=str(cif_path), is_u_affected=is_u,
                        n_atoms=-1, converged=False, n_steps=0,
                        initial_loss=float("nan"),
                        final_loss=float("nan"),
                        best_loss=float("nan"),
                        fmax_initial=float("nan"),
                        fmax_final=float("nan"),
                        min_d_initial=float("nan"),
                        min_d_final=float("nan"),
                        mean_M_O_bond=None,
                        runtime_sec=time.time() - t0,
                        error=err,
                    )
                    print(f"    ✗ {err}")
                summary_rows.append(asdict(traj))

        # Free the MACE calculator between teachers — they each take ~5 GB.
        del calc
        if MACE_DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

    # ── Write summary CSV ────────────────────────────────────────────────
    summary_path = OUTPUT_ROOT / "summary.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\n[summary] wrote {summary_path}")
    else:
        print(f"\n[summary] no trajectories produced — nothing to write")

    # ── Quick text report ────────────────────────────────────────────────
    print("\n=== Quick comparison (mean over seeds) ===")
    print(f"{'system':<8} {'is_U':<5} {'teacher':<6} "
          f"{'final_loss':>12} {'fmax_final':>10} {'mean_M-O':>9} "
          f"{'converged':>9}")
    by_key = {}
    for row in summary_rows:
        k = (row["system"], row["teacher"])
        by_key.setdefault(k, []).append(row)
    seen_sys = []
    for (sys_label, *_), _ in [(m, None) for m in TEST_MATERIALS]:
        for teacher_label in TEACHERS:
            rows = by_key.get((sys_label, teacher_label), [])
            if not rows:
                continue
            mean_final_loss = np.nanmean([r["final_loss"]   for r in rows])
            mean_fmax       = np.nanmean([r["fmax_final"]   for r in rows])
            mean_MO_vals    = [r["mean_M_O_bond"] for r in rows
                                if r["mean_M_O_bond"] is not None]
            mean_MO = np.mean(mean_MO_vals) if mean_MO_vals else float("nan")
            n_conv          = sum(1 for r in rows if r["converged"])
            is_u = "yes" if rows[0]["is_u_affected"] else "no"
            print(f"{sys_label:<8} {is_u:<5} {teacher_label:<6} "
                  f"{mean_final_loss:>12.3e} {mean_fmax:>10.3e} "
                  f"{mean_MO:>9.3f} {n_conv:>2}/{len(rows)}")
        seen_sys.append(sys_label)


if __name__ == "__main__":
    main()
