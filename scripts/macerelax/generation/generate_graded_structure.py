"""Generate a single structure with a spatial disorder gradient, then
relax it with MACE+wall.

Vision (see MACE_RELAX_PILOT.md): a material elongated along one axis whose
packing is initialised as a continuous spectrum from amorphous to crystalline,
then relaxed *as one structure* so the final geometry follows a smooth
disordered → ordered → disordered spectrum along the long axis.

Pipeline:
  1. tricor variable-density graded packing (Supercell.generate_graded,
     num_steps=0) in an anisotropic [CS, CS, LONG] box.  Grain size + per-grain
     crystallinity vary along the long axis via a period-1 cosine profile, so
     the cell's long-axis wrap joins like-to-like (symmetric-periodic).
  2. bond_relax geometric cleanup (NOT shell_relax) — heals Voronoi-boundary
     overlaps without erasing the gradient.
  3. MACE-MPA + min-distance wall, FIRE, N_STEPS steps.  Uses the SAME FIRE
     params as the big_v1 generator (generate_mace_trajectories.py):
     maxstep=0.3, N_STEPS=60.  Every frame is saved (stride 1).

       *** Watch the gradient vs step count. ***
     Per MACE_RELAX_PILOT.md §2g, MACE amplifies the order gradient for the
     first ~10 steps then tends to collapse regimes toward a common glassy
     basin as the relaxation deepens.  We do NOT pre-empt that by under-
     relaxing — we relax with the standard params and let
     analyze_graded_structure.py scan the saved frames (SCAN_STEPS) to find
     where the spatial gradient is strongest and where it starts to wash out.

  4. save NPZ trajectory + extxyz frames (initial / cleaned / final), each
     carrying a per-atom `order` coordinate (0=disordered, 1=ordered) for
     colouring and per-slab analysis (see analyze_graded_structure.py).

This script does NOT run automatically as part of any pipeline — run it
explicitly on a GPU node:

    python scripts/macerelax/generation/generate_graded_structure.py
"""
from __future__ import annotations

# Thread caps, set BEFORE numpy/torch imports so they take effect.
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "8")

import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from ase.optimize import FIRE
from mace.calculators import mace_mp

torch.set_num_threads(4)

import tricor as tc

sys.path.insert(0, str(Path(__file__).parent))
from wall_calculator import MinDistanceWallCalculator, per_pair_min_from_atoms


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# --- Reference crystal ---
REFERENCE_CIF = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos/mp-7000_SiO2.cif")
SYSTEM_LABEL  = "SiO2_quartz_graded"   # used for output filenames

# --- Geometry (anisotropic box: long axis gets the gradient) ---
# 25 x 25 x 200 Å at quartz density (~0.075 atoms/Å³ @ rel_density 0.94) is
# ~9.4k atoms — comfortably inside the validated MACE-MPA H100 envelope
# (the pilot OOM'd at ~14-15k; see MACE_RELAX_PILOT.md §10.1).  Bump the box
# only if you confirm headroom, or port the OOM auto-shrink loop first.
CROSS_SECTION_ANGSTROMS = 25.0    # the two short axes (Å)
LONG_AXIS_ANGSTROMS     = 200.0   # the elongated axis (Å) — gradient runs here
LONG_AXIS               = 2        # 0=x, 1=y, 2=z
RELATIVE_DENSITY        = 0.94     # HELD CONSTANT along the axis (order-only gradient)
RNG_SEED                = 12345

# --- Gradient shape ---
ORDER_PROFILE        = "cosine_disordered_ends"  # disordered ends, ordered core
# Gradient is driven SOLELY by grain size, exactly like tricor's real regime
# ladder: every preset from `amorphous` (grain_size=6) to `crystalline_30`
# (grain_size=30) packs at crystalline_fraction=1.0 — the disorder comes from
# grain SIZE (tiny randomly-oriented crystallites → amorphous-like), not from
# amorphous-filled grains.  So crystalline_prob is pinned to 1.0 here; the
# crystalline_prob_* knobs (not a real tricor packing param) stay inert.
GRAIN_SIZE_MIN       = 6.0    # = tricor amorphous preset grain_size (disordered end)
GRAIN_SIZE_MAX       = 30.0   # = tricor crystalline_30 grain_size (ordered end)
CRYST_PROB_MIN       = 1.0    # all grains crystalline (matches the regime ladder)
CRYST_PROB_MAX       = 1.0    # all grains crystalline
CRYST_PROB_GAMMA     = 1.0    # inert when min==max
DISPLACEMENT_SIGMA   = 0.0    # thermal jitter (Å); amorphous preset uses 0.08 but
                              # the big_v1 MACE pipeline zeroes it — we match that.

# --- Geometric overlap cleanup (before MACE) ---
BOND_RELAX_N_ITER    = 80
BOND_RELAX_MAX_STEP  = 0.1

# --- MACE + wall relaxation ---
MACE_MODEL           = "medium-mpa-0"
MACE_DEVICE          = "cuda"
MACE_DEFAULT_DTYPE   = "float32"
N_STEPS              = 60      # MATCHES big_v1 generate_mace_trajectories.py.
                               # We SAVE EVERY FRAME (stride 1); FIRE evolves
                               # deterministically, so the frames at any step in
                               # this single trajectory ARE that step's result —
                               # analyze_graded_structure.py slices them
                               # (SCAN_STEPS) to find where the spatial gradient
                               # is strongest and where it collapses (§2g).
OPT_MAXSTEP          = 0.3     # FIRE maxstep (Å) — MATCHES big_v1 (was 0.1, which
                               # under-relaxed: atoms barely moved, fmax ~unchanged).
FMAX_TARGET          = 0.05    # may not be reached in 60 steps; N_STEPS is the cap
WALL_K               = 1000.0
WALL_EXPONENT        = 4
WALL_MARGIN          = 0.0

# --- Output ---
OUTPUT_DIR = Path("/home/ehrdt/tricor/mace/data/graded_v1")

# ══════════════════════════════════════════════════════════════════════════════


def build_graded_pack():
    """tricor graded variable-density packing (no shell_relax FIRE)."""
    ref = ase_read(str(REFERENCE_CIF), format="cif")
    cell_dim = [CROSS_SECTION_ANGSTROMS, CROSS_SECTION_ANGSTROMS,
                CROSS_SECTION_ANGSTROMS]
    cell_dim[LONG_AXIS] = LONG_AXIS_ANGSTROMS

    cell = tc.Supercell.from_atoms(
        ref,
        cell_dim_angstroms=cell_dim,
        relative_density=RELATIVE_DENSITY,
        rng_seed=RNG_SEED,
        label=SYSTEM_LABEL,
    )
    shell = tc.CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)
    summary = cell.generate_graded(
        shell,
        long_axis=LONG_AXIS,
        order_profile=ORDER_PROFILE,
        grain_size_min=GRAIN_SIZE_MIN,
        grain_size_max=GRAIN_SIZE_MAX,
        crystalline_prob_min=CRYST_PROB_MIN,
        crystalline_prob_max=CRYST_PROB_MAX,
        crystalline_prob_gamma=CRYST_PROB_GAMMA,
        displacement_sigma=DISPLACEMENT_SIGMA,
        num_steps=0,
        show_progress=False,
    )
    return cell, shell, summary


def run_mace_relax(atoms, base_calc):
    """FIRE on the MACE+wall PES, saving every step.  Wall thresholds are
    derived per-run from the bond_relax-cleaned structure.  FIRE is the
    optimizer used by the big_v1 trajectory generator (it is memory-less,
    so each step's displacement is a clean function of the current state)."""
    r_min_per_pair = per_pair_min_from_atoms(atoms, margin=WALL_MARGIN)
    atoms.calc = MinDistanceWallCalculator(
        base_calc=base_calc, r_min_per_pair=r_min_per_pair,
        k=WALL_K, exponent=WALL_EXPONENT,
    )

    snapshots: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    energy_history: list[float] = []

    e0 = float(atoms.get_potential_energy())
    f0 = atoms.get_forces()
    fmax_initial = float(np.abs(f0).max())
    snapshots.append(atoms.positions.copy().astype(np.float32))
    snapshot_steps.append(0)
    energy_history.append(e0)

    opt = FIRE(atoms, maxstep=OPT_MAXSTEP, logfile=None)

    def per_step_callback():
        energy_history.append(float(atoms.get_potential_energy()))
        snapshots.append(atoms.positions.copy().astype(np.float32))
        snapshot_steps.append(int(opt.nsteps))

    opt.attach(per_step_callback, interval=1)
    opt.run(fmax=FMAX_TARGET, steps=N_STEPS)

    if snapshot_steps[-1] != int(opt.nsteps):
        snapshots.append(atoms.positions.copy().astype(np.float32))
        snapshot_steps.append(int(opt.nsteps))

    return {
        "positions":      np.stack(snapshots, axis=0),
        "snapshot_steps": np.asarray(snapshot_steps, dtype=np.int32),
        "loss":           np.asarray(energy_history, dtype=np.float64),
        "fmax_initial":   fmax_initial,
        "fmax_final":     float(np.abs(atoms.get_forces()).max()),
        "n_opt_steps":    int(opt.nsteps),
        "wall_thresholds": r_min_per_pair,
    }


def _write_xyz(path, numbers, positions, cell, order):
    a = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    a.set_array("order", np.asarray(order, dtype=np.float64))
    ase_write(str(path), a, format="extxyz")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Graded-structure generation  ({SYSTEM_LABEL})")
    print(f"  reference: {REFERENCE_CIF}")
    print(f"  box: cross={CROSS_SECTION_ANGSTROMS} Å  long(axis {LONG_AXIS})="
          f"{LONG_AXIS_ANGSTROMS} Å  rel_density={RELATIVE_DENSITY}")
    print(f"  gradient: {ORDER_PROFILE}  grain {GRAIN_SIZE_MIN}->{GRAIN_SIZE_MAX} Å  "
          f"P_cryst {CRYST_PROB_MIN}->{CRYST_PROB_MAX} (gamma={CRYST_PROB_GAMMA})")
    print(f"  MACE: {MACE_MODEL}  FIRE steps={N_STEPS} maxstep={OPT_MAXSTEP} "
          f"(matches big_v1)\n")

    # ---- 1+2. graded pack + cleanup ----
    t0 = time.perf_counter()
    cell, shell, summary = build_graded_pack()
    order = cell.graded_order_coordinate()
    numbers = cell.atoms.numbers.astype(np.int32).copy()
    box = np.asarray(cell.atoms.cell.array, dtype=np.float32)
    initial_positions = cell.atoms.positions.astype(np.float32).copy()
    print(f"  packed: atoms={summary['num_atoms']}  grains={summary['n_grains']}  "
          f"crystalline={summary['n_crystalline_grains']} "
          f"(frac={summary['crystalline_fraction']:.2f})  "
          f"actual_density={summary['actual_density']}  "
          f"{time.perf_counter()-t0:.1f}s")
    _write_xyz(OUTPUT_DIR / f"{SYSTEM_LABEL}_initial.xyz",
               numbers, initial_positions, box, order)

    cell.bond_relax(shell, n_iter=BOND_RELAX_N_ITER, max_step=BOND_RELAX_MAX_STEP)
    cleaned_positions = cell.atoms.positions.astype(np.float32).copy()
    # order is a function of the (fixed) long-axis coordinate; recompute on the
    # cleaned positions so atoms that drifted across slab edges are re-labelled.
    order_cleaned = cell.graded_order_coordinate()
    _write_xyz(OUTPUT_DIR / f"{SYSTEM_LABEL}_cleaned.xyz",
               numbers, cleaned_positions, box, order_cleaned)
    print(f"  bond_relax done  {time.perf_counter()-t0:.1f}s")

    # ---- 3. MACE + wall ----
    print("  initializing MACE (cold start ~30s)...")
    calc = mace_mp(model=MACE_MODEL, device=MACE_DEVICE,
                   default_dtype=MACE_DEFAULT_DTYPE)
    h = run_mace_relax(cell.atoms, calc)
    final_positions = cell.atoms.positions.astype(np.float32).copy()
    order_final = cell.graded_order_coordinate()
    _write_xyz(OUTPUT_DIR / f"{SYSTEM_LABEL}_final.xyz",
               numbers, final_positions, box, order_final)
    print(f"  MACE relax: {h['n_opt_steps']} steps  "
          f"E {h['loss'][0]:.1f}->{h['loss'][-1]:.1f}  "
          f"fmax {h['fmax_initial']:.2f}->{h['fmax_final']:.2f}  "
          f"{time.perf_counter()-t0:.1f}s")

    # ---- 4. save trajectory NPZ ----
    wall_thresholds = h["wall_thresholds"]
    out_npz = OUTPUT_DIR / f"{SYSTEM_LABEL}_trajectory.npz"
    np.savez(
        out_npz,
        positions=h["positions"],
        snapshot_steps=h["snapshot_steps"],
        initial_positions=initial_positions,
        cleaned_positions=cleaned_positions,
        final_positions=final_positions,
        species_numbers=numbers,
        cell=box,
        order=order.astype(np.float32),
        loss_history=h["loss"],
        long_axis=np.int32(LONG_AXIS),
        order_profile=np.asarray(ORDER_PROFILE),
        grain_size_min=np.float32(GRAIN_SIZE_MIN),
        grain_size_max=np.float32(GRAIN_SIZE_MAX),
        cryst_prob_min=np.float32(CRYST_PROB_MIN),
        cryst_prob_max=np.float32(CRYST_PROB_MAX),
        cryst_prob_gamma=np.float32(CRYST_PROB_GAMMA),
        rel_density=np.float32(RELATIVE_DENSITY),
        n_grains=np.int32(summary["n_grains"]),
        n_crystalline_grains=np.int32(summary["n_crystalline_grains"]),
        fmax_initial=np.float32(h["fmax_initial"]),
        fmax_final=np.float32(h["fmax_final"]),
        num_steps=np.int32(h["n_opt_steps"]),
        system_label=np.asarray(SYSTEM_LABEL),
        reference_cif=np.asarray(str(REFERENCE_CIF)),
        backend=np.asarray("mace+wall"),
        optimizer=np.asarray("FIRE"),
        mace_model=np.asarray(MACE_MODEL),
    )
    size_mb = out_npz.stat().st_size / (1024 * 1024)

    # Also dump the full trajectory as a multi-frame extxyz so it opens
    # directly in OVITO (animation), coloured by the per-atom `order` column.
    from npz_trajectory_to_xyz import npz_to_trajectory_xyz
    out_traj_xyz = OUTPUT_DIR / f"{SYSTEM_LABEL}_trajectory.xyz"
    n_frames = npz_to_trajectory_xyz(out_npz, out_traj_xyz)

    print(f"\nDone in {(time.perf_counter()-t0)/60:.1f} min")
    print(f"  trajectory: {out_npz}  ({size_mb:.1f} MB)")
    print(f"  trajectory (OVITO): {out_traj_xyz}  ({n_frames} frames, colour by 'order')")
    print(f"  xyz frames: {SYSTEM_LABEL}_{{initial,cleaned,final}}.xyz in {OUTPUT_DIR}")
    print(f"  next: python scripts/macerelax/generation/analyze_graded_structure.py")


if __name__ == "__main__":
    main()
