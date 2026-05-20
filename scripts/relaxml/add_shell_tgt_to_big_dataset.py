"""Retrofit shell_target arrays into the relaxml_big_v1 trajectories.

Specialized variant of ``add_shell_tgt_to_npz.py`` for the big-dataset
layout produced by ``generate_big_dataset.py``:

  * Per-CIF subdirs under ``DATASET_ROOT`` (recursive glob).
  * Each .npz already carries ``compound`` and ``mp_id`` fields, so the
    CIF lookup is exact (``{mp_id}_{compound}.cif`` in CIF_DIR) and
    works correctly even when one compound has many polymorphs.
  * Contexts cached by ``(compound, mp_id)`` so multi-polymorph
    compounds don't collide.
  * Parallel across NUM_WORKERS processes; files are sorted so each
    worker tends to see a run of same-CIF files and hits its cache.

Idempotent: a file with ``shell_pair_species`` already in its keys is
skipped.  Atomic per-file rewrite via ``.tmp.npz`` + rename, so a
crash mid-run leaves no partial files.
"""

from __future__ import annotations

# Limit BLAS threads before numpy import.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path("/wigeon/users/ehrdt/prod/relaxml_big_v1")
CIF_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV_training")

# Workers process files in parallel.  Retrofit is mostly file I/O (no
# heavy compute besides 516 one-time CoordinationShellTarget builds, each
# cached per (compound, mp_id) inside each worker), so I/O bandwidth
# tends to dominate.  Keep this <= the generator's worker count to avoid
# RAM pressure on the shared box (see RELAXML_SESSION.txt §19.6).
NUM_WORKERS = 12

# Set True to re-derive shell_target arrays even when a file already has
# them.  Use after fixing a bug in the extractor.
OVERWRITE = False

# Sweep .tmp.npz orphans from prior crashed runs before starting.  The
# atomic rewrite below uses ``.tmp.npz`` which always renames on success;
# leftovers only happen if the process was killed mid-rewrite.
CLEAN_TMP_ORPHANS = True

# ─────────────────────────────────────────────────────────────────────────────


# Per-worker module-level cache.  Populated lazily on first use of a
# given (compound, mp_id).  Keys are tuples → values are dicts of
# numpy arrays (shell_pair_species, shell_pair_features,
# shell_triplet_species, shell_triplet_features).
_CTX_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] = {}
_CIF_DIR: Path | None = None


def _init_worker(cif_dir_str: str) -> None:
    """multiprocessing.Pool initializer — runs once per worker process."""
    global _CIF_DIR
    _CIF_DIR = Path(cif_dir_str)


def _get_context(compound: str, mp_id: str) -> dict[str, np.ndarray]:
    """Lazy per-worker cache: shell_target arrays for one (compound, mp_id)."""
    key = (compound, mp_id)
    cached = _CTX_CACHE.get(key)
    if cached is not None:
        return cached

    # Lazy imports — heavy ase / tricor modules; only one worker needs to
    # pay this cost per unique CIF.
    from ase.io import read as ase_read
    from tricor.relaxml.shell_target import extract_shell_target_arrays
    from tricor.shells import CoordinationShellTarget

    cif_path = _CIF_DIR / f"{mp_id}_{compound}.cif"
    if not cif_path.is_file():
        raise FileNotFoundError(
            f"CIF not found: {cif_path}.  Expected layout: "
            f"{{mp_id}}_{{compound}}.cif under {_CIF_DIR}.  Check that "
            f"the .npz's compound={compound!r} mp_id={mp_id!r} matches "
            f"a file in the training CIF dir."
        )
    ref = ase_read(str(cif_path), format="cif")
    target = CoordinationShellTarget.from_atoms(ref)
    arrays = extract_shell_target_arrays(target)
    _CTX_CACHE[key] = arrays
    return arrays


def _process_file(path_str: str) -> tuple[str, str]:
    """Process one .npz.  Returns (status, info) where status is
    'added' | 'skipped' | 'failed' and info is the failure reason
    (empty for non-failures).
    """
    path = Path(path_str)
    try:
        with np.load(path) as npz:
            if (not OVERWRITE) and ("shell_pair_species" in npz.files):
                return ("skipped", "")
            if "compound" not in npz.files or "mp_id" not in npz.files:
                return ("failed", f"{path.name}: missing compound/mp_id key")
            compound = str(npz["compound"].item())
            mp_id = str(npz["mp_id"].item())
            payload = {
                k: npz[k] for k in npz.files
                if k not in (
                    "shell_pair_species", "shell_pair_features",
                    "shell_triplet_species", "shell_triplet_features",
                )
            }

        arrays = _get_context(compound, mp_id)
        payload.update(arrays)

        # Atomic rewrite: write to a sibling .tmp.npz, then rename onto
        # the target.  Resolve symlinks first so we modify the real file.
        target = path.resolve()
        tmp = target.with_name(target.stem + ".tmp.npz")
        try:
            np.savez(tmp, **payload)
            shutil.move(str(tmp), str(target))
        except Exception:
            if tmp.is_file():
                tmp.unlink()
            raise
        return ("added", "")
    except Exception as e:
        return ("failed", f"{path.name}: {type(e).__name__}: {e}")


def _sweep_tmp_orphans(root: Path) -> int:
    """Delete leftover ``*.tmp.npz`` files from crashed runs.  These
    confuse downstream globs and are useless (the original .npz is
    intact because rename is atomic)."""
    orphans = list(root.rglob("*.tmp.npz"))
    for o in orphans:
        try:
            o.unlink()
        except OSError:
            pass
    return len(orphans)


def main() -> None:
    if not DATASET_ROOT.is_dir():
        raise SystemExit(f"DATASET_ROOT not found: {DATASET_ROOT}")
    if not CIF_DIR.is_dir():
        raise SystemExit(f"CIF_DIR not found: {CIF_DIR}")

    if CLEAN_TMP_ORPHANS:
        n_orphans = _sweep_tmp_orphans(DATASET_ROOT)
        if n_orphans:
            print(f"[cleanup] removed {n_orphans} stale .tmp.npz files")

    print(f"Globbing .npz under {DATASET_ROOT}/ ...")
    npz_files = sorted(DATASET_ROOT.rglob("*.npz"))
    # Defensive: exclude any .tmp.npz that escaped the orphan sweep.
    npz_files = [p for p in npz_files if not p.name.endswith(".tmp.npz")]
    if not npz_files:
        raise SystemExit(f"No .npz files under {DATASET_ROOT}")
    print(f"Found {len(npz_files)} .npz files")
    print(f"Workers: {NUM_WORKERS}.  CIF dir: {CIF_DIR}")
    print()

    n_added = n_skipped = n_failed = 0
    failures: list[str] = []
    t0 = time.perf_counter()

    # imap with chunksize sends batches of consecutive files to each
    # worker.  Because npz_files is sorted (per-CIF subdirs cluster
    # together), each worker tends to see a run of same-CIF files and
    # hits its (compound, mp_id) cache after the first one.
    with mp.Pool(
        NUM_WORKERS, initializer=_init_worker, initargs=(str(CIF_DIR),),
    ) as pool:
        total = len(npz_files)
        for i, (status, info) in enumerate(
            pool.imap_unordered(
                _process_file, [str(p) for p in npz_files], chunksize=8,
            ),
            start=1,
        ):
            if status == "added":
                n_added += 1
            elif status == "skipped":
                n_skipped += 1
            else:
                n_failed += 1
                failures.append(info)
            if i % 500 == 0 or i == total:
                dt = time.perf_counter() - t0
                rate = i / dt if dt > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{total}] added={n_added} skipped={n_skipped} "
                    f"failed={n_failed}  ({dt:.0f}s, {rate:.1f}/s, "
                    f"ETA {eta:.0f}s)",
                    flush=True,
                )

    dt = time.perf_counter() - t0
    print()
    print(
        f"Done in {dt:.0f}s.  added={n_added}  skipped={n_skipped}  "
        f"failed={n_failed}  total={n_added + n_skipped + n_failed}"
    )
    if failures:
        print(f"\nFirst {min(20, len(failures))} failure messages:")
        for f in failures[:20]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
