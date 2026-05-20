"""Phase-contrast trajectory generator for SiO₂ polymorphs.

This is the data-side fix for "model ignores shell_target" (see
RELAXML_SESSION.txt Part 13.6).  Standard training trajectories give
the model a single ``(positions, shell_target) → relaxation`` pair
per starting structure, so initial positions can carry enough
information to determine the polymorph and shell_target becomes
redundant.  Phase contrast forces the model to use shell_target by
producing matched pairs:

    (positions_X, shell_target_A) → relaxation_toward_polymorph_A
    (positions_X, shell_target_B) → relaxation_toward_polymorph_B

— same starting positions, different shell_targets, different
relaxation outcomes.  If the model wants to fit both rows, it has
to actually use shell_target.

Algorithm
---------
Outer loop over polymorphs (mirrors the multiprocessing pattern in
``generate_sio2_trajectories_v2.py``).  For each polymorph:

  * Spin up a ``multiprocessing.Pool`` of ``NUM_WORKERS`` workers.
  * The pool initializer caches the polymorph's reference + shell_target
    and the BASE reference (used for Supercell construction) in
    per-worker globals so they're built once per process.
  * Each task is a single ``start_idx``.  Worker calls
    ``Supercell.from_atoms(base_ref, rng_seed=BASE_SEED+start_idx)``
    so the pre-disorder positions are reproducible *across polymorphs*
    for a given start_idx, then ``sc.generate(polymorph_shell_target,
    ...)`` runs the liquid PRESET shell_relax with the polymorph's
    coordination/distance/angle targets.

Result: ``N_LIQUID_STARTS × len(POLYMORPHS)`` trajectories in
``OUT_DIR``, each with the polymorph's shell_target arrays embedded
into the .npz so the existing RelaxMLDataModule + evaluate.py work
unchanged.

Notes
-----
* The BASE reference (α-quartz) is used only for Supercell
  construction — cell shape, atom count, G3Distribution measurement.
  All polymorphs are SiO₂ so composition is identical.
* ``rng_seed = BASE_SEED + start_idx`` is shared across polymorphs
  for a given start_idx, so ``Supercell.from_atoms`` produces
  identical pre-disorder positions for all 5 polymorph runs at that
  start_idx.  ``sc.generate`` may apply slightly different
  pre-separation per polymorph (each polymorph has its own
  ``pair_hard_min``), but this is bounded by ~0.35 * hard_min ≈ 0.5 Å
  and is much smaller than the actual relaxation motion.
* Quality control (verifying that each shell_relax actually reached
  the polymorph) is left to a separate script
  ``filter_phase_contrast_qc.py``.  Some liquid → stishovite
  transitions (random → 6-coord octahedral) may fail to converge.
  This generator saves everything; QC filters after.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from pathlib import Path

# --- resource caps ---
GPU_ID = 0
NUM_THREADS = 4           # per worker, not total
NUM_WORKERS = 8           # parallel processes; set <=1 for serial

# --- corpus shape ---
N_LIQUID_STARTS = 100
CELL_SIZE = 50.0          # Å — cubic supercell edge
REL_DENSITY = 0.96
BASE_SEED = 700_000       # rng_seed = BASE_SEED + start_idx
TRAJECTORY_STRIDE = 5     # save every 5th step into positions[]

# --- polymorphs ---
# All SiO2 polymorphs to generate per liquid start.  BASE_CIF is used
# as the reference for Supercell construction (cell shape, atom count,
# G3Distribution measurement); per-polymorph shell_targets come from
# each polymorph's own CIF and are passed to ``generate`` to drive
# the relaxation.
CIF_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos")
BASE_CIF = "mp-7000_SiO2.cif"           # α-quartz, 4-coord, used as Supercell base


@dataclass(frozen=True)
class PolymorphSpec:
    tag: str
    cif_filename: str


POLYMORPHS: list[PolymorphSpec] = [
    PolymorphSpec("alpha_quartz",       "mp-7000_SiO2.cif"),
    PolymorphSpec("alpha_cristobalite", "mp-6945_SiO2.cif"),
    PolymorphSpec("beta_cristobalite",  "mp-546794_SiO2.cif"),
    PolymorphSpec("coesite",            "mp-6930_SiO2.cif"),
    PolymorphSpec("stishovite",         "mp-6947_SiO2.cif"),
]

# --- shell_relax kwargs ---
# Use Supercell.PRESETS["liquid"] for the relaxation.  This is a soft
# spring network with bond_weight=0.4, angle_weight=0.5 — gives the
# atoms freedom to rearrange substantially.  The polymorph-specific
# shell_target tells it WHERE to relax toward.
PRESET_NAME = "liquid"

# --- output ---
COMPOUND_NAME = "SiO2"
OUT_DIR = Path(__file__).parent / "data" / "phase_contrast_v1" / "SiO2"
MANIFEST_NAME = "manifest.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Apply resource caps BEFORE importing numpy/torch.  Workers inherit
# these via the spawn context.
# ─────────────────────────────────────────────────────────────────────────────

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
_n = str(NUM_THREADS)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = _n

import csv
import multiprocessing as mp
import sys
import time

import numpy as np
from ase.io import read as ase_read

from tricor.shells import CoordinationShellTarget
from tricor.supercell import Supercell
from tricor.relaxml.shell_target import extract_shell_target_arrays


# Stable column order for the manifest.  Mirrors the columns in
# scripts/relaxml/generate_surrogate_trajectories_multicomp.py so the
# existing RelaxMLDataModule manifest loader works unchanged.  Extra
# phase-contrast metadata (``start_idx``, ``polymorph``) appended at
# the end — the loader ignores unrecognized columns.
MANIFEST_COLUMNS = [
    "idx", "compound", "source", "regime", "polymorph",
    "rng_seed",
    "grain_size", "num_grains", "n_crystalline", "crystalline_fraction",
    "bond_weight", "angle_weight", "repulsion_weight",
    "hard_core_scale", "nonbond_push_scale", "displacement_sigma",
    "num_steps", "cell_size", "rel_density",
    "initial_loss", "best_loss", "final_loss",
    "runtime_s", "filename",
    "start_idx",
]

# ─────────────────────────────────────────────────────────────────────────────
# Worker setup — globals filled by _init_worker, consumed by _run_one
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_BASE_REF = None
_WORKER_SHELL_TARGET = None
_WORKER_POLYMORPH_TAG = ""
_WORKER_GEN_KWARGS: dict = {}
_WORKER_NUM_STEPS = 0
_WORKER_OUT_DIR: "Path | None" = None
_WORKER_FLAT_IDX_OFFSET = 0


def _shell_relax_kwargs_from_preset(preset: dict) -> tuple[dict, int]:
    """Strip ``num_steps`` from a copy of the preset and return
    ``(generate_kwargs, num_steps)``."""
    p = dict(preset)
    num_steps = int(p.pop("num_steps", 100))
    return p, num_steps


def _init_worker(
    base_cif_str: str,
    polymorph_tag: str,
    polymorph_cif_str: str,
    gen_kwargs: dict,
    num_steps: int,
    out_dir_str: str,
    flat_idx_offset: int,
) -> None:
    """Pool initializer.  Builds the base reference + this polymorph's
    shell_target once per worker process and caches them in globals."""
    global _WORKER_BASE_REF, _WORKER_SHELL_TARGET, _WORKER_POLYMORPH_TAG
    global _WORKER_GEN_KWARGS, _WORKER_NUM_STEPS
    global _WORKER_OUT_DIR, _WORKER_FLAT_IDX_OFFSET

    _WORKER_BASE_REF = ase_read(base_cif_str)
    poly_ref = ase_read(polymorph_cif_str)
    _WORKER_SHELL_TARGET = CoordinationShellTarget.from_atoms(poly_ref)
    _WORKER_POLYMORPH_TAG = polymorph_tag
    _WORKER_GEN_KWARGS = dict(gen_kwargs)
    _WORKER_NUM_STEPS = int(num_steps)
    _WORKER_OUT_DIR = Path(out_dir_str)
    _WORKER_FLAT_IDX_OFFSET = int(flat_idx_offset)


def _run_one(start_idx: int) -> "dict | None":
    """Entry point for each pool task.  Returns a manifest-row dict
    (or None on failure) for the ``(start_idx, _WORKER_POLYMORPH_TAG)``
    pair."""
    try:
        return run_trajectory(start_idx)
    except Exception as e:
        print(
            f"[start_idx={start_idx:04d} poly={_WORKER_POLYMORPH_TAG}] "
            f"FAILED: {type(e).__name__}: {e}",
            flush=True,
        )
        return None


def run_trajectory(start_idx: int) -> dict:
    """Run one phase-contrast trajectory in the calling process.

    Pulls the polymorph data from the worker globals so the costly
    CoordinationShellTarget construction happens once per worker, not
    once per task.  Sequential / main-process callers should set the
    globals manually before calling this.
    """
    assert _WORKER_BASE_REF is not None, "_init_worker not called"
    assert _WORKER_SHELL_TARGET is not None, "_init_worker not called"
    assert _WORKER_OUT_DIR is not None, "_init_worker not called"

    rng_seed = BASE_SEED + start_idx
    flat_idx = _WORKER_FLAT_IDX_OFFSET + start_idx

    t0 = time.perf_counter()
    sc = Supercell.from_atoms(
        _WORKER_BASE_REF,
        cell_dim_angstroms=CELL_SIZE,
        rng_seed=rng_seed,
        relative_density=REL_DENSITY,
    )

    summary = sc.generate(
        _WORKER_SHELL_TARGET,
        num_steps=_WORKER_NUM_STEPS,
        save_trajectory=True,
        trajectory_stride=TRAJECTORY_STRIDE,
        show_progress=False,
        **_WORKER_GEN_KWARGS,
    )
    runtime = time.perf_counter() - t0

    h = sc.shell_relax_history
    filename = (
        f"phase_contrast_start{start_idx:04d}_{_WORKER_POLYMORPH_TAG}"
        f"_seed{rng_seed:09d}.npz"
    )
    outfile = _WORKER_OUT_DIR / filename

    shell_target_arrays = extract_shell_target_arrays(_WORKER_SHELL_TARGET)

    np.savez(
        outfile,
        positions=h["positions"],
        snapshot_steps=h["snapshot_steps"],
        initial_positions=np.asarray(
            sc.atoms.positions, dtype=np.float32,
        ),  # NB: post-relax positions; for true initial state use
            # positions[0].  Kept for schema parity with the v2
            # multicomp generator.
        best_positions=h["best_positions"],
        final_positions=sc.atoms.positions.astype(np.float32),
        species_numbers=sc.atoms.numbers.astype(np.int32),
        cell=np.asarray(sc.atoms.cell.array, dtype=np.float32),
        loss_history=h["loss"],
        **shell_target_arrays,
        idx=np.int64(flat_idx),
        source=np.asarray("phase_contrast"),
        regime=np.asarray(PRESET_NAME),
        compound=np.asarray(COMPOUND_NAME),
        polymorph=np.asarray(_WORKER_POLYMORPH_TAG),
        rng_seed=np.int64(rng_seed),
        start_idx=np.int32(start_idx),
        grain_size=np.float32(0.0),
        num_grains=np.int32(0),
        n_crystalline=np.int32(0),
        crystalline_fraction=np.float32(0.0),
        bond_weight=np.float32(_WORKER_GEN_KWARGS.get("bond_weight", 1.0)),
        angle_weight=np.float32(_WORKER_GEN_KWARGS.get("angle_weight", 0.5)),
        repulsion_weight=np.float32(_WORKER_GEN_KWARGS.get("repulsion_weight", 3.0)),
        hard_core_scale=np.float32(_WORKER_GEN_KWARGS.get("hard_core_scale", 1.0)),
        nonbond_push_scale=np.float32(_WORKER_GEN_KWARGS.get("nonbond_push_scale", 1.0)),
        displacement_sigma=np.float32(_WORKER_GEN_KWARGS.get("displacement_sigma", 0.0)),
        num_steps=np.int32(_WORKER_NUM_STEPS),
        trajectory_stride=np.int32(TRAJECTORY_STRIDE),
        cell_size=np.float32(CELL_SIZE),
        rel_density=np.float32(REL_DENSITY),
        initial_loss=np.float64(summary["initial_loss"]),
        best_loss=np.float64(summary["best_loss"]),
        final_loss=np.float64(summary["final_loss"]),
    )

    size_mb = outfile.stat().st_size / (1024 * 1024)
    print(
        f"[{flat_idx+1:5d}] start={start_idx:04d} "
        f"poly={_WORKER_POLYMORPH_TAG:>18s} atoms={len(sc.atoms):5d} "
        f"loss {summary['initial_loss']:6.2f}→{summary['final_loss']:6.2f} "
        f"best={summary['best_loss']:6.2f}  "
        f"{runtime:5.1f}s  {size_mb:5.1f}MB",
        flush=True,
    )

    return {
        "idx": flat_idx,
        "compound": COMPOUND_NAME,
        "source": "phase_contrast",
        "regime": PRESET_NAME,
        "polymorph": _WORKER_POLYMORPH_TAG,
        "rng_seed": rng_seed,
        "grain_size": 0.0,
        "num_grains": 0,
        "n_crystalline": 0,
        "crystalline_fraction": 0.0,
        "bond_weight":          _WORKER_GEN_KWARGS.get("bond_weight", 1.0),
        "angle_weight":         _WORKER_GEN_KWARGS.get("angle_weight", 0.5),
        "repulsion_weight":     _WORKER_GEN_KWARGS.get("repulsion_weight", 3.0),
        "hard_core_scale":      _WORKER_GEN_KWARGS.get("hard_core_scale", 1.0),
        "nonbond_push_scale":   _WORKER_GEN_KWARGS.get("nonbond_push_scale", 1.0),
        "displacement_sigma":   _WORKER_GEN_KWARGS.get("displacement_sigma", 0.0),
        "num_steps": _WORKER_NUM_STEPS,
        "cell_size": CELL_SIZE,
        "rel_density": REL_DENSITY,
        "initial_loss": float(summary["initial_loss"]),
        "best_loss": float(summary["best_loss"]),
        "final_loss": float(summary["final_loss"]),
        "runtime_s": float(runtime),
        "filename": filename,
        "start_idx": start_idx,
    }


def main() -> None:
    if not CIF_DIR.is_dir():
        sys.exit(f"CIF_DIR does not exist: {CIF_DIR}")
    base_cif = CIF_DIR / BASE_CIF
    if not base_cif.is_file():
        sys.exit(f"BASE_CIF not found: {base_cif}")
    for ps in POLYMORPHS:
        cif_path = CIF_DIR / ps.cif_filename
        if not cif_path.is_file():
            sys.exit(f"Polymorph CIF not found: {cif_path}")

    preset = dict(Supercell.PRESETS[PRESET_NAME])
    preset.pop("relative_density", None)
    gen_kwargs, num_steps = _shell_relax_kwargs_from_preset(preset)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / MANIFEST_NAME

    total_runs = N_LIQUID_STARTS * len(POLYMORPHS)
    print(
        f"phase-contrast generation: {N_LIQUID_STARTS} liquid starts × "
        f"{len(POLYMORPHS)} polymorphs = {total_runs} trajectories"
    )
    print(f"  out_dir : {OUT_DIR}")
    print(f"  base_cif: {BASE_CIF}")
    print(f"  preset  : '{PRESET_NAME}'  num_steps={num_steps}")
    print(f"  cell    : {CELL_SIZE} Å  rel_density={REL_DENSITY}")
    print(f"  workers : {NUM_WORKERS}")
    print()

    all_rows: list[dict] = []
    overall_t0 = time.perf_counter()

    for poly_idx, ps in enumerate(POLYMORPHS):
        polymorph_cif = str(CIF_DIR / ps.cif_filename)
        flat_idx_offset = poly_idx * N_LIQUID_STARTS
        configs = list(range(N_LIQUID_STARTS))

        print(f"=== polymorph {poly_idx+1}/{len(POLYMORPHS)}: {ps.tag} ===")

        if NUM_WORKERS is None or NUM_WORKERS <= 1:
            # Serial path — mostly useful for debugging since spawn
            # overhead isn't a thing here.
            _init_worker(
                str(base_cif), ps.tag, polymorph_cif,
                gen_kwargs, num_steps,
                str(OUT_DIR), flat_idx_offset,
            )
            for start_idx in configs:
                row = _run_one(start_idx)
                if row is not None:
                    all_rows.append(row)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=NUM_WORKERS,
                initializer=_init_worker,
                initargs=(
                    str(base_cif), ps.tag, polymorph_cif,
                    gen_kwargs, num_steps,
                    str(OUT_DIR), flat_idx_offset,
                ),
            ) as pool:
                for row in pool.imap_unordered(_run_one, configs, chunksize=1):
                    if row is not None:
                        all_rows.append(row)

        # Re-write manifest after each polymorph completes so a crash
        # mid-run leaves a usable partial manifest.
        if all_rows:
            with open(manifest_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row)

    total_runtime = time.perf_counter() - overall_t0
    print()
    print(
        f"Done: wrote {len(all_rows)}/{total_runs} trajectories in "
        f"{total_runtime / 3600:.2f}h "
        f"({total_runtime / max(len(all_rows), 1):.1f}s/trajectory)"
    )
    print(f"  manifest: {manifest_path}")
    print(
        f"  Run filter_phase_contrast_qc.py next to flag trajectories that "
        f"failed to relax toward their target polymorph."
    )


if __name__ == "__main__":
    main()
