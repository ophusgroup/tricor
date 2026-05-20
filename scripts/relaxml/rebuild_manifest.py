"""Rebuild a per-compound manifest.csv from the .npz files in a trajectory dir.

The multi-species generator writes manifest.csv in ``"w"`` (overwrite) mode
on every run.  If you re-ran the generator (e.g. to add more samples) the
old run's .npz files persist on disk but the manifest gets clobbered to
only describe the latest run — leaving N orphan .npz files with no
manifest rows.  Symptom: ``merge.py`` reports more .npz files than
manifest rows after concatenation.

This script rebuilds manifest.csv by reading every .npz file in the
target directory(ies) and reconstructing each row from the metadata
keys saved by the generator.  The only field that *isn't* recoverable
from the .npz is ``runtime_s`` — we preserve it from the existing
manifest where the row still exists, and use NaN otherwise.  Training
doesn't use ``runtime_s``; it's monitoring metadata only.

Usage:
    python scripts/relaxml/rebuild_manifest.py <traj_dir> [<traj_dir> ...]

For each directory, this writes <dir>/manifest.csv.tmp, validates that
every .npz produced a row, then atomically renames over the old
manifest.  The old manifest is moved to manifest.csv.bak first so it's
recoverable if anything goes wrong.

Example to clean up the SiC and SiO2 dirs that got 500 + 149 orphan
npz files from re-runs:

    python scripts/relaxml/rebuild_manifest.py \\
        ./scripts/relaxml/data/multi_species_v1/SiC_trajectories \\
        ./scripts/relaxml/data/multi_species_v1/SiO2_trajectories

Then re-run merge.py — row count should match .npz count.
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
from pathlib import Path

import numpy as np


# Manifest columns, in the order the generator writes them.  Source of
# truth: scripts/relaxml/generate_surrogate_trajectories_multicomp.py
# `run_trajectory` return dict (around line 555).
MANIFEST_COLUMNS = (
    "idx",
    "compound",
    "source",
    "regime",
    "rng_seed",
    "grain_size",
    "num_grains",
    "n_crystalline",
    "crystalline_fraction",
    "bond_weight",
    "angle_weight",
    "repulsion_weight",
    "hard_core_scale",
    "nonbond_push_scale",
    "displacement_sigma",
    "num_steps",
    "initial_loss",
    "best_loss",
    "final_loss",
    "num_atoms",
    "runtime_s",
    "filename",
)


def _scalar(npz, key, cast):
    """Read a 0-d array from npz and cast to a python scalar."""
    v = npz[key]
    # 0-d arrays unwrap via .item(); strings come out as numpy bytes.
    try:
        return cast(v.item())
    except Exception:
        return cast(v)


def _compound_from_dirname(traj_dir: Path) -> str:
    """Infer compound name from the trajectory directory.

    Convention from the generator: ``DATASET_ROOT/<COMPOUND>_trajectories/``.
    """
    name = traj_dir.name
    if name.endswith("_trajectories"):
        return name[: -len("_trajectories")]
    return name  # fall back to the dir name as-is


def _row_from_npz(npz_path: Path, compound: str,
                  prior_runtime: dict[str, float]) -> dict:
    """Build one manifest row by reading metadata keys from an .npz file.

    `prior_runtime` is a {filename → runtime_s} dict from the existing
    manifest (where rows exist for this file); we preserve it.  For
    orphan .npz files (no prior row) we use NaN.
    """
    with np.load(npz_path, allow_pickle=False) as npz:
        try:
            n_atoms = int(npz["species_numbers"].shape[0])
        except KeyError:
            raise ValueError(f"{npz_path.name}: missing species_numbers")
        # Required metadata fields — every field that's stored as a
        # 0-d array in the npz.  Strings come back as numpy bytes; we
        # decode with .item() then cast to str.
        try:
            row = {
                "idx":                  _scalar(npz, "idx",                  int),
                "compound":             compound,
                "source":               _scalar(npz, "source",               str),
                "regime":               _scalar(npz, "regime",               str),
                "rng_seed":             _scalar(npz, "rng_seed",             int),
                "grain_size":           _scalar(npz, "grain_size",           float),
                "num_grains":           _scalar(npz, "num_grains",           int),
                "n_crystalline":        _scalar(npz, "n_crystalline",        int),
                "crystalline_fraction": _scalar(npz, "crystalline_fraction", float),
                "bond_weight":          _scalar(npz, "bond_weight",          float),
                "angle_weight":         _scalar(npz, "angle_weight",         float),
                "repulsion_weight":     _scalar(npz, "repulsion_weight",     float),
                "hard_core_scale":      _scalar(npz, "hard_core_scale",      float),
                "nonbond_push_scale":   _scalar(npz, "nonbond_push_scale",   float),
                "displacement_sigma":   _scalar(npz, "displacement_sigma",   float),
                "num_steps":            _scalar(npz, "num_steps",            int),
                "initial_loss":         _scalar(npz, "initial_loss",         float),
                "best_loss":            _scalar(npz, "best_loss",            float),
                "final_loss":           _scalar(npz, "final_loss",           float),
                "num_atoms":            n_atoms,
                "runtime_s":            prior_runtime.get(npz_path.name, math.nan),
                "filename":             npz_path.name,
            }
        except KeyError as e:
            raise ValueError(
                f"{npz_path.name}: missing metadata key {e!s} — was this "
                f".npz produced by an old generator version that didn't "
                f"save full metadata?"
            )
    return row


def _load_prior_runtime(manifest_path: Path) -> dict[str, float]:
    """Pull existing {filename → runtime_s} so we preserve runtime
    measurements from rows that still exist.  Tolerant of missing
    file or columns."""
    if not manifest_path.is_file():
        return {}
    out: dict[str, float] = {}
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            return {}
        for row in reader:
            fn = row.get("filename")
            rt = row.get("runtime_s")
            if not fn:
                continue
            try:
                out[fn] = float(rt) if rt not in (None, "") else math.nan
            except ValueError:
                out[fn] = math.nan
    return out


def rebuild_one(traj_dir: Path) -> tuple[int, int]:
    """Rebuild manifest.csv for a single trajectory directory.

    Returns (n_npz, n_rows_written).  Writes to manifest.csv.tmp, then
    moves the old manifest to manifest.csv.bak and renames the new
    file into place.
    """
    if not traj_dir.is_dir():
        raise SystemExit(f"Not a directory: {traj_dir}")
    npz_files = sorted(traj_dir.glob("*.npz"))
    if not npz_files:
        print(f"  {traj_dir}: no .npz files, nothing to rebuild")
        return 0, 0

    compound = _compound_from_dirname(traj_dir)
    manifest_path = traj_dir / "manifest.csv"
    prior_runtime = _load_prior_runtime(manifest_path)

    rows: list[dict] = []
    n_failed = 0
    for npz_path in npz_files:
        try:
            row = _row_from_npz(npz_path, compound, prior_runtime)
            rows.append(row)
        except Exception as e:
            print(f"  [skip] {npz_path.name}: {type(e).__name__}: {e}")
            n_failed += 1

    if not rows:
        raise SystemExit(
            f"  {traj_dir}: no .npz file produced a valid row — refusing "
            f"to overwrite manifest"
        )

    # Stable ordering: idx ascending, then rng_seed.  Matches the order
    # the generator would have produced if it had run all configs in
    # one pass.
    rows.sort(key=lambda r: (r["idx"], r["rng_seed"]))

    tmp_path = traj_dir / "manifest.csv.tmp"
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Backup the old manifest if it exists, then atomic rename.
    if manifest_path.is_file():
        bak = traj_dir / "manifest.csv.bak"
        shutil.move(str(manifest_path), str(bak))
    tmp_path.rename(manifest_path)

    n_recovered = sum(1 for r in rows if not math.isnan(r["runtime_s"]))
    n_orphan = len(rows) - n_recovered
    print(
        f"  {traj_dir.name}: wrote {len(rows)} rows from {len(npz_files)} "
        f".npz files (skipped {n_failed}, runtime_s preserved on "
        f"{n_recovered}, NaN on {n_orphan})"
    )
    return len(npz_files), len(rows)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    targets = [Path(a).resolve() for a in sys.argv[1:]]
    print(f"Rebuilding manifests for {len(targets)} directory(ies):")
    totals = (0, 0)
    for d in targets:
        n_npz, n_rows = rebuild_one(d)
        totals = (totals[0] + n_npz, totals[1] + n_rows)
    print()
    print(f"Total: {totals[0]} .npz files, {totals[1]} manifest rows written")
    if totals[0] != totals[1]:
        print(f"WARNING: total rows ({totals[1]}) != total .npz ({totals[0]}) "
              f"— some .npz files were skipped due to errors")


if __name__ == "__main__":
    main()
