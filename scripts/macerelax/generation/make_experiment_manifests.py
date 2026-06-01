"""Emit per-experiment train/eval manifests plus the complete master.

Replaces the earlier mace/rebuild_manifest.py.  Walks every .npz under
DATASET_ROOT, reads the embedded metadata, then writes:

  - DATASET_ROOT / manifest_all.csv
        Complete record of every NPZ on disk, regardless of experiment.
        Audit + analysis source of truth.

  - DATASET_ROOT / manifests / <exp>_train.csv
  - DATASET_ROOT / manifests / <exp>_eval.csv
        Per-experiment filtered manifests.  The training data loader
        (RelaxMLDataModule) does a random 90/10 train/val split over its
        input manifest, so feeding it the <exp>_train.csv ensures
        validation samples are in-distribution and eval-held-out systems
        are never seen during training.

  - DATASET_ROOT / manifests / experiments.json
        Registry mapping experiment name → manifest paths, system lists,
        trajectory counts, and a generation timestamp.  Train scripts
        resolve EXPERIMENT_NAME against this registry so the only thing
        a user changes is the experiment name.

EXPERIMENTS below is the canonical declaration of what each experiment
trains on and what it holds out.  Add new experiments by appending an
entry and re-running this script.  Idempotent — re-running overwrites
the per-experiment manifests + registry, never touches the .npz files.

Filenames in the emitted manifests are paths RELATIVE to DATASET_ROOT
(e.g. `train/SiC/SiC_SRO_..._.npz`) because the data loader resolves
files via `data_root / row["filename"]` with
`data_root = manifest_path.parent`.  The master manifest sits at
DATASET_ROOT, per-experiment manifests sit one level deeper at
DATASET_ROOT/manifests/.  The script writes the appropriate relative
prefix for each (so the per-experiment manifests use `../train/...`).

Run with:
    /home/ehrdt/miniforge3/envs/mace/bin/python scripts/macerelax/generation/make_experiment_manifests.py
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# === CONFIG ============================================================
DATASET_ROOT = Path("/home/ehrdt/tricor/mace/data/pilot_v1")
MANIFESTS_SUBDIR = DATASET_ROOT / "manifests"

# Per-experiment train/eval system lists.  Add new entries here over time;
# each becomes a pair of manifests + a registry entry.
EXPERIMENTS: dict[str, dict] = {
    "pilot_v1_full": {
        "description": (
            "Full pilot v1: all train-role systems for training; all "
            "eval-role systems held out.  Use this when you want a single "
            "model scored against every cross-axis."
        ),
        "train_systems": [
            "Si", "SiC",
            "SiO2_quartz", "SiO2_cristobalite1", "SiO2_cristobalite2",
            "AlN", "BN",
        ],
        "eval_systems": [
            "Si3N4", "SiO2_coesite", "SiO2_stishovite",
        ],
    },
    "composition_test": {
        "description": (
            "Cross-chemistry generalization.  Trains on Si + SiC + 3 SiO2 "
            "polymorphs + AlN + h-BN; holds out Si3N4.  The two nitrides "
            "give the model N-containing training signal (AlN: 4-coord N, "
            "h-BN: 3-coord N like Si3N4) so the held-out Si3N4 evaluation "
            "tests cross-cation generalization with N coordination chemistry "
            "in-distribution, instead of asking the model to extrapolate to "
            "an unseen element."
        ),
        "train_systems": [
            "Si", "SiC",
            "SiO2_quartz", "SiO2_cristobalite1", "SiO2_cristobalite2",
            "AlN", "BN",
        ],
        "eval_systems": ["Si3N4"],
    },
    "polymorph_test": {
        "description": (
            "Cross-polymorph generalization within SiO2.  Trains on three "
            "4-coordinate SiO2 polymorphs (quartz + 2 cristobalite); holds "
            "out coesite (4-coord, different lattice) and stishovite "
            "(6-coord — different coordination, the OOD stress test).  "
            "Deliberately excludes Si and SiC to isolate cross-polymorph "
            "from cross-chemistry effects."
        ),
        "train_systems": [
            "SiO2_quartz", "SiO2_cristobalite1", "SiO2_cristobalite2",
        ],
        "eval_systems": [
            "SiO2_coesite", "SiO2_stishovite",
        ],
    },
}

# Output column order.  Must include every column the macerelax data loader
# reads from the manifest (see src/tricor/macerelax/data.py).
COLUMNS = (
    "system_id", "role", "composition", "mp_id",
    "idx", "regime", "rng_seed",
    "n_atoms",
    "grain_size", "num_grains", "crystalline_fraction", "rel_density",
    "wall_global_min", "fmax_initial",
    "initial_loss", "best_loss", "final_loss",
    "fmax_final", "num_opt_steps",
    "filename",  # path relative to the manifest's containing dir
)
# =======================================================================


def _read_npz_metadata(npz_path: Path) -> dict:
    """Extract a manifest row from a single NPZ (filename left empty;
    callers fill it in per-manifest because the relative path depends on
    where the manifest sits)."""
    with np.load(npz_path) as d:
        files = set(d.files)

        def get(key, default=None):
            if key not in files:
                return default
            return d[key].item()

        return {
            "system_id":            str(get("system_id", "")),
            "role":                 str(get("role", "")),
            "composition":          str(get("composition", "")),
            "mp_id":                str(get("mp_id", "")),
            "idx":                  int(get("idx", 0)),
            "regime":               str(get("regime", "")),
            "rng_seed":             int(get("rng_seed", 0)),
            "n_atoms":              int(d["species_numbers"].shape[0]),
            "grain_size":           float(get("grain_size", 0.0)),
            "num_grains":           int(get("num_grains", 0)),
            "crystalline_fraction": float(get("crystalline_fraction", 0.0)),
            "rel_density":          float(get("rel_density", 0.0)),
            "wall_global_min":      float(get("wall_global_min", 0.0)),
            "fmax_initial":         float(get("fmax_initial", 0.0)),
            "initial_loss":         float(get("initial_loss", 0.0)),
            "best_loss":            float(get("best_loss", 0.0)),
            "final_loss":           float(get("final_loss", 0.0)),
            "fmax_final":           float(get("fmax_final", 0.0)),
            "num_opt_steps":        int(get("num_steps", 0)),
            "_abs_path":            npz_path,  # for relative-path computation
        }


def _write_csv(out_path: Path, rows: list[dict], rel_root: Path) -> int:
    """Write one CSV with `filename` set as the relative path from
    rel_root to each row's `_abs_path`.  Returns the row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in COLUMNS if k != "filename"}
            row["filename"] = os.path.relpath(r["_abs_path"], rel_root)
            writer.writerow(row)
    return len(rows)


def main():
    npz_files = sorted(DATASET_ROOT.rglob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files under {DATASET_ROOT}")

    all_rows: list[dict] = []
    failures: list[tuple[Path, str]] = []
    for path in npz_files:
        try:
            all_rows.append(_read_npz_metadata(path))
        except Exception as e:
            failures.append((path, repr(e)))

    print(f"  read OK:      {len(all_rows)}")
    print(f"  read FAILED:  {len(failures)}")
    for p, msg in failures[:5]:
        print(f"    {p.name}: {msg}")

    # Stable sort for diff-friendly output.
    all_rows.sort(key=lambda r: (r["role"], r["system_id"], r["idx"],
                                  r["regime"]))

    # Sanity: per-system row counts.
    per_system: dict[str, int] = {}
    for r in all_rows:
        per_system[r["system_id"]] = per_system.get(r["system_id"], 0) + 1
    print("\nPer-system NPZ counts:")
    for sid in sorted(per_system):
        print(f"  {sid:25s}  {per_system[sid]:>4d}")

    # ── 1. Master manifest at DATASET_ROOT ────────────────────────────
    master_path = DATASET_ROOT / "manifest_all.csv"
    n_master = _write_csv(master_path, all_rows, rel_root=DATASET_ROOT)
    print(f"\nWrote master: {n_master:>4d} rows  → {master_path}")

    # ── 2. Per-experiment manifests + registry ────────────────────────
    MANIFESTS_SUBDIR.mkdir(parents=True, exist_ok=True)
    all_system_ids = set(per_system.keys())

    registry: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_npz_root": str(DATASET_ROOT),
        "source_npz_count": len(all_rows),
        "experiments": {},
    }

    print()
    for exp_name, spec in EXPERIMENTS.items():
        train_systems = set(spec["train_systems"])
        eval_systems = set(spec["eval_systems"])
        # Sanity: experiment must only reference systems that exist on disk.
        missing = (train_systems | eval_systems) - all_system_ids
        if missing:
            print(f"[warn] {exp_name}: unknown system_ids {sorted(missing)} "
                  f"— skipping")
            continue
        # Sanity: a system cannot be both train and eval in one experiment.
        overlap = train_systems & eval_systems
        if overlap:
            raise ValueError(
                f"{exp_name}: systems {sorted(overlap)} appear in BOTH "
                f"train_systems and eval_systems — fix EXPERIMENTS config."
            )

        train_rows = [r for r in all_rows if r["system_id"] in train_systems]
        eval_rows  = [r for r in all_rows if r["system_id"] in eval_systems]

        train_path = MANIFESTS_SUBDIR / f"{exp_name}_train.csv"
        eval_path  = MANIFESTS_SUBDIR / f"{exp_name}_eval.csv"
        n_train = _write_csv(train_path, train_rows, rel_root=MANIFESTS_SUBDIR)
        n_eval  = _write_csv(eval_path,  eval_rows,  rel_root=MANIFESTS_SUBDIR)

        registry["experiments"][exp_name] = {
            "description":             spec["description"],
            "train_systems":           sorted(spec["train_systems"]),
            "eval_systems":            sorted(spec["eval_systems"]),
            "train_manifest":          train_path.name,
            "eval_manifest":           eval_path.name,
            "n_train_trajectories":    n_train,
            "n_eval_trajectories":     n_eval,
        }
        print(f"  {exp_name:20s}  train={n_train:>4d}  eval={n_eval:>4d}")

    registry_path = MANIFESTS_SUBDIR / "experiments.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")
    print(f"\nRegistry: {registry_path}")

    print("\nTo train on an experiment, set in scripts/macerelax/train.py:")
    print(f"  EXPERIMENT_NAME = \"<one of: {', '.join(EXPERIMENTS)}>\"")
    print(f"  EXPERIMENTS_REGISTRY = \"{registry_path}\"")


if __name__ == "__main__":
    main()
