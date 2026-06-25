"""Build a flat ``dataset_table`` — one row per trajectory.

Scans ``OUTPUT_ROOT`` and joins every persistent data source into a single
spreadsheet-friendly table:

  * ``OUTPUT_ROOT/_runs/*.json``           — run manifests (config + git + slurm)
  * ``OUTPUT_ROOT/<sys>_generated/manifest.csv``  — per-CIF trajectory rows
  * ``OUTPUT_ROOT/_runs/{run_id}.rank*.csv``      — per-rank summaries
  * ``OUTPUT_ROOT/summary.csv``            — legacy pre-augmentation summary (optional)
  * ``OUTPUT_ROOT/enrichment.csv``         — per-traj scalars from enrich_metadata.py (optional)

Each row gets the CONFIG snapshot from its ``run_id`` exploded into prefixed
columns (``run_base_seed``, ``run_cell_dims``, ``run_use_bf16_inference``, …).
Output:

  * ``OUTPUT_ROOT/dataset_table.parquet``   — columnar, ~10× smaller than CSV,
    types preserved, queryable via duckdb/pandas without loading into memory
  * ``OUTPUT_ROOT/dataset_table.csv``       — human-readable copy for Excel/etc.

Idempotent — rebuilds from on-disk state every run.  Safe to run mid-corpus
to peek at progress.  Cheap (pure pandas) — runs on a login node.

Usage:
    python scripts/macerelax/build_dataset_table.py
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

OUTPUT_ROOT = Path("/pscratch/sd/e/ehrdt/macerelax/generated_v1")

# Where the materialized table is written.
DATASET_TABLE_PARQUET = OUTPUT_ROOT / "dataset_table.parquet"
DATASET_TABLE_CSV     = OUTPUT_ROOT / "dataset_table.csv"

# Optional sources — controlled by flags so a fresh tree without enrichment
# still produces a valid table.
INCLUDE_LEGACY_SUMMARY = True   # OUTPUT_ROOT/summary.csv from pre-augmentation runs
INCLUDE_ENRICHMENT     = True   # OUTPUT_ROOT/enrichment.csv from enrich_metadata.py

# Prefix used when exploding the per-run CONFIG snapshot into per-row
# columns.  Keeps run-level vs trajectory-level columns visually distinct.
CONFIG_PREFIX = "run_"

# When True, print a per-regime breakdown + OOM/failure counts after writing.
PRINT_STATS = True

# ─────────────────────────────────────────────────────────────────────────────

import json
import sys

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Run manifests → flat dict of per-run columns
# ─────────────────────────────────────────────────────────────────────────────

# Top-level run-JSON fields that get promoted as scalar columns.
_RUN_TOPLEVEL_FIELDS = (
    "run_id", "schema_version", "status",
    "started_at_utc", "ended_at_utc",
)


def _flatten_run_payload(payload: dict) -> dict:
    """Flatten a run JSON to a single-row dict of prefixed scalars.

    Nested dicts (slurm, git, model, config) are flattened one level deep.
    Lists/dicts inside ``config`` (e.g. ``DENSITY_BY_REGIME``) become JSON
    strings so the column stays scalar (parquet + pandas both handle that
    fine and you can ``json.loads`` per row if you want the dict back).
    """
    out: dict = {}
    for k in _RUN_TOPLEVEL_FIELDS:
        out[f"{CONFIG_PREFIX}{k}"] = payload.get(k)
    for section in ("slurm", "git", "model"):
        sub = payload.get(section) or {}
        for k, v in sub.items():
            out[f"{CONFIG_PREFIX}{section}_{k}"] = v
    cfg = payload.get("config") or {}
    for k, v in cfg.items():
        col = f"{CONFIG_PREFIX}{k.lower()}"
        if isinstance(v, (list, tuple, dict)):
            out[col] = json.dumps(v)
        else:
            out[col] = v
    # Surface end-of-run stats as columns too — useful for "which run died?"
    stats = payload.get("stats") or {}
    for k, v in stats.items():
        out[f"{CONFIG_PREFIX}stats_{k}"] = v
    return out


def _load_run_manifests(runs_dir: Path) -> dict[str, dict]:
    """run_id → flattened payload dict.  Skips unparseable / runid-less files."""
    runs: dict[str, dict] = {}
    if not runs_dir.is_dir():
        return runs
    for path in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"[skip] {path.name}: {exc}", file=sys.stderr)
            continue
        rid = payload.get("run_id") or path.stem
        runs[rid] = _flatten_run_payload(payload)
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory row sources
# ─────────────────────────────────────────────────────────────────────────────

# Canonical key for trajectory identity within a run.  ``cif_filename`` rather
# than ``cif_idx`` because a row might come from a legacy summary without the
# new ``cif_idx`` column.
_TRAJ_KEY_COLS = ("run_id", "cif_filename", "regime", "rng_seed")


def _load_per_cif_manifests(root: Path) -> pd.DataFrame:
    """Concat every ``<sys>_generated/manifest.csv`` into one frame."""
    frames: list[pd.DataFrame] = []
    for cif_dir in sorted(root.glob("*_generated")):
        man = cif_dir / "manifest.csv"
        if not man.is_file():
            continue
        try:
            df = pd.read_csv(man)
        except Exception as exc:
            print(f"[skip] {man}: {exc}", file=sys.stderr)
            continue
        df["_source"] = "per_cif"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_per_rank_summaries(runs_dir: Path) -> pd.DataFrame:
    """Concat every ``_runs/*.rank*.csv`` into one frame."""
    frames: list[pd.DataFrame] = []
    if not runs_dir.is_dir():
        return pd.DataFrame()
    for path in sorted(runs_dir.glob("*.rank*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        df["_source"] = "per_rank"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_legacy_summary(root: Path) -> pd.DataFrame:
    """Read ``OUTPUT_ROOT/summary.csv`` (pre-augmentation runs) if present."""
    path = root / "summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[skip] {path}: {exc}", file=sys.stderr)
        return pd.DataFrame()
    # Legacy rows have no run_id — fill with NaN so the run-info join is a
    # no-op for them.  They still show up in the table with their core
    # fields populated.
    if "run_id" not in df.columns:
        df["run_id"] = pd.NA
    df["_source"] = "legacy_summary"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Combine + join
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_within_source(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate keys within one source — keep the LAST occurrence.

    Last-wins because per-CIF manifest.csv is append-only: if a CIF was
    re-run, the more recent row is the one we want.
    """
    if df.empty:
        return df
    key_cols = [c for c in _TRAJ_KEY_COLS if c in df.columns]
    if not key_cols:
        return df
    return df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)


def _combine_sources(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple row sources, deduplicating by ``_TRAJ_KEY_COLS``.

    Earlier-listed frames win on conflict.  Pass the most authoritative
    source first: per-CIF manifest (live, incremental) > per-rank summary
    (end-of-run snapshot) > legacy summary (pre-augmentation, no run_id).
    """
    frames = tuple(_dedup_within_source(f) for f in frames if not f.empty)
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]

    combined = frames[0]
    key_cols = [c for c in _TRAJ_KEY_COLS if c in combined.columns]
    for next_frame in frames[1:]:
        if not key_cols or not all(c in next_frame.columns for c in key_cols):
            # Schema mismatch — append all rows; downstream dedup is a no-op.
            combined = pd.concat([combined, next_frame], ignore_index=True)
            continue
        marker = next_frame.merge(
            combined[key_cols].assign(_already=True),
            on=key_cols, how="left",
        )
        extras = marker.loc[marker["_already"].isna()].drop(columns=["_already"])
        combined = pd.concat([combined, extras], ignore_index=True)
    return combined


def _join_with_runs(rows: pd.DataFrame, runs: dict[str, dict]) -> pd.DataFrame:
    """Left-join the per-row table with the per-run flattened payloads."""
    if rows.empty or not runs:
        return rows
    runs_df = pd.DataFrame.from_dict(runs, orient="index")
    # The flattened dict already has its own ``run_run_id`` column; the join
    # key is just ``run_id`` from the trajectory side.
    runs_df = runs_df.reset_index().rename(columns={"index": "run_id"})
    if "run_id" not in rows.columns:
        return rows
    return rows.merge(runs_df, on="run_id", how="left", suffixes=("", "_runinfo"))


def _load_glob_frames(root: Path, glob_pattern: str) -> pd.DataFrame:
    """Concat every CSV matching ``glob_pattern`` into one frame.

    Used for both ``enrichment*.csv`` (cheap pass) and
    ``mace_enrichment*.csv`` (MACE pass), which may exist as either a
    single shared file or per-rank shards.  Deduplicates by ``source_file``
    keeping the last occurrence.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(glob_pattern)):
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"[skip] {path.name}: {exc}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "source_file" in df.columns:
        df = df.drop_duplicates("source_file", keep="last")
    return df


def _join_enrichment(rows: pd.DataFrame, root: Path,
                     glob_pattern: str, suffix: str) -> pd.DataFrame:
    """Left-join the trajectory rows with an enrichment frame.

    Match on whichever subset of (run_id, cif_filename, regime, rng_seed) is
    present in both.  Conflicting non-key columns get ``suffix`` appended
    (e.g., ``_enr``, ``_mace``).
    """
    if rows.empty:
        return rows
    enr = _load_glob_frames(root, glob_pattern)
    if enr.empty:
        return rows
    keys = [c for c in _TRAJ_KEY_COLS if c in enr.columns and c in rows.columns]
    if not keys:
        print(f"[warn] {glob_pattern} has no join keys; skipping",
              file=sys.stderr)
        return rows
    return rows.merge(enr, on=keys, how="left", suffixes=("", suffix))


# ─────────────────────────────────────────────────────────────────────────────
# Output + stats
# ─────────────────────────────────────────────────────────────────────────────

def _write_outputs(df: pd.DataFrame) -> None:
    """Write parquet (if pyarrow/fastparquet available) + CSV."""
    DATASET_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(DATASET_TABLE_PARQUET, index=False)
        print(f"[write] {DATASET_TABLE_PARQUET}  "
              f"({len(df):,} rows × {len(df.columns)} cols)")
    except Exception as exc:
        print(f"[warn] parquet write failed ({exc}); CSV-only "
              f"(install pyarrow for parquet)", file=sys.stderr)
    df.to_csv(DATASET_TABLE_CSV, index=False)
    print(f"[write] {DATASET_TABLE_CSV}  "
          f"({len(df):,} rows × {len(df.columns)} cols)")


def _print_corpus_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("(empty table)")
        return
    print()
    print("─── corpus summary ─────────────────────────────────────────")
    print(f"  total rows         : {len(df):,}")
    if "error" in df.columns:
        n_ok = int((df["error"].fillna("") == "").sum())
        n_fail = len(df) - n_ok
        print(f"  succeeded          : {n_ok:,}")
        print(f"  failed             : {n_fail:,}")
        if n_fail:
            fail_types = (
                df.loc[df["error"].fillna("") != "", "error"]
                .value_counts().head(8)
            )
            for t, c in fail_types.items():
                print(f"    {c:>5d} × {str(t)[:60]}")
    if "regime" in df.columns:
        print("  per-regime counts:")
        for reg, c in df["regime"].value_counts().sort_index().items():
            print(f"    {str(reg):<20s} {c:>6d}")
    if "run_id" in df.columns:
        n_runs = df["run_id"].dropna().nunique()
        print(f"  distinct run_ids   : {n_runs}")
        if f"{CONFIG_PREFIX}status" in df.columns:
            statuses = (
                df.drop_duplicates("run_id")
                .set_index("run_id")[f"{CONFIG_PREFIX}status"]
                .value_counts()
                .to_dict()
            )
            for s, c in statuses.items():
                print(f"    {str(s):<20s} {c:>3d} run(s)")
    if "n_atoms" in df.columns:
        n = pd.to_numeric(df["n_atoms"], errors="coerce")
        n = n[n > 0]
        if not n.empty:
            print(f"  n_atoms range      : {int(n.min()):,} – "
                  f"{int(n.max()):,}  (median {int(n.median()):,})")
    if "peak_gpu_gb" in df.columns:
        v = pd.to_numeric(df["peak_gpu_gb"], errors="coerce")
        v = v[v > 0]
        if not v.empty:
            print(f"  peak_gpu_gb median : {v.median():.1f}  "
                  f"(max {v.max():.1f})")
    if "runtime_sec" in df.columns:
        v = pd.to_numeric(df["runtime_sec"], errors="coerce")
        v = v[v > 0]
        if not v.empty:
            print(f"  runtime_sec median : {v.median():.1f}  "
                  f"(p95 {v.quantile(0.95):.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not OUTPUT_ROOT.is_dir():
        sys.exit(f"[abort] OUTPUT_ROOT does not exist: {OUTPUT_ROOT}")
    runs_dir = OUTPUT_ROOT / "_runs"

    print(f"[scan] OUTPUT_ROOT = {OUTPUT_ROOT}", flush=True)
    runs = _load_run_manifests(runs_dir)
    print(f"[scan] {len(runs)} run manifest(s) under {runs_dir.name}/")

    per_cif = _load_per_cif_manifests(OUTPUT_ROOT)
    print(f"[scan] {len(per_cif):,} rows from per-CIF manifests")

    per_rank = _load_per_rank_summaries(runs_dir)
    print(f"[scan] {len(per_rank):,} rows from per-rank summaries")

    legacy = _load_legacy_summary(OUTPUT_ROOT) if INCLUDE_LEGACY_SUMMARY \
             else pd.DataFrame()
    if not legacy.empty:
        print(f"[scan] {len(legacy):,} rows from legacy summary.csv")

    combined = _combine_sources(per_cif, per_rank, legacy)
    print(f"[join] {len(combined):,} unique trajectories after dedup")

    joined = _join_with_runs(combined, runs)

    if INCLUDE_ENRICHMENT:
        enr_files = sorted(OUTPUT_ROOT.glob("enrichment*.csv"))
        if enr_files:
            joined = _join_enrichment(joined, OUTPUT_ROOT,
                                       "enrichment*.csv", "_enr")
            print(f"[join] {len(enr_files)} enrichment file(s) joined "
                  f"({len(joined):,} rows)")
        else:
            print("[scan] no enrichment*.csv yet (skipped)")

        mace_files = sorted(OUTPUT_ROOT.glob("mace_enrichment*.csv"))
        if mace_files:
            joined = _join_enrichment(joined, OUTPUT_ROOT,
                                       "mace_enrichment*.csv", "_mace")
            print(f"[join] {len(mace_files)} mace_enrichment file(s) joined "
                  f"({len(joined):,} rows)")
        else:
            print("[scan] no mace_enrichment*.csv yet (skipped)")

    # Stable row order so diffs across rebuilds are minimal.
    sort_cols = [c for c in ("run_id", "cif_idx", "cif_filename", "regime",
                              "rng_seed") if c in joined.columns]
    if sort_cols:
        joined = joined.sort_values(sort_cols).reset_index(drop=True)

    # The _source helper column is for our internal dedup; not useful in
    # the final table.
    if "_source" in joined.columns:
        joined = joined.drop(columns=["_source"])

    _write_outputs(joined)
    if PRINT_STATS:
        _print_corpus_summary(joined)


if __name__ == "__main__":
    main()
