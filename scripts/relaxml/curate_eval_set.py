"""Curate a per-system OOD evaluation set by copying selected CIFs
from the rest-dir generation into a self-contained eval root.

Each system gets its own subdir with all 20 trajectories + per-system
manifest.csv.  The eval script (``eval_systems.py``) later iterates
these subdirs.

Also retrofits each copied .npz with shell_target arrays (required by
the dataloader at eval time) if ``RETROFIT=True``.  The retrofit uses
the same machinery as ``add_shell_tgt_to_big_dataset.py``: looks up
``{mp_id}_{compound}.cif`` in CIF_DIR and writes shell_pair / shell_triplet
arrays into the .npz.

The curated set is OOD against the 516-CIF training subset:
  Axis 1 — cross-polymorph: composition present in training, different mp_id
  Axis 2 — cross-composition: at least one element pair never paired in training

Edit the CONFIG block below, then run:
    python scripts/relaxml/curate_eval_set.py
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import shutil
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_ROOT = Path("/wigeon/users/ehrdt/prod/relaxml_big_rest")
CIF_DIR     = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV")
TARGET_ROOT = Path("/wigeon/users/ehrdt/prod/eval_set_v1")

# Whether to retrofit shell_target arrays into the copied .npz files.
# Required for the dataloader to load these at eval time.  Idempotent —
# files that already have shell_pair_species are skipped.
RETROFIT = True

# Curated eval systems.  (compound_formula, mp_id) — the corresponding
# subdir under SOURCE_ROOT must be named ``{compound}_{mp_id}_trajectories``.
# Molecular-solid crystals (CO2, CCl4, etc.) intentionally excluded —
# their van-der-Waals-dominated bonding is too far from the training
# distribution of network/ionic/intermetallic solids to give a useful
# OOD signal.
SYSTEMS: list[tuple[str, str]] = [
    # ── Axis 1: cross-polymorph (formula seen in training, different mp_id) ──
    ("Ag2S",  "mp-1095694"),
    ("Ag2S",  "mp-1102900"),
    ("BaO",   "mp-1008500"),
    ("C",     "mp-1040425"),
    ("Cr3N2", "mp-1014303"),

    # ── Axis 2: cross-composition with a new element pair ──
    #     Ordered easy → hard by min element coverage in training:
    #     ZnO (Zn:18), Rb2O (Rb:12), Nb2O5 (Nb:11) are positive
    #     controls — well-trained elements never paired in training.
    #     AgO (Ag:4), Bi2O3 (Bi:4) are the harder regime where pair
    #     generalization has to compensate for thin metal-side coverage.
    ("ZnO",   "mp-1093993"),
    ("Rb2O",  "mp-1101405"),
    ("Nb2O5", "mp-1101660"),
    ("AgO",   "mp-1065190"),
    ("Bi2O3", "mp-1017552"),
]

# ─────────────────────────────────────────────────────────────────────────────


def _subdir_name(compound: str, mp_id: str) -> str:
    return f"{compound}_{mp_id}_trajectories"


def _copy_system(compound: str, mp_id: str) -> tuple[int, int]:
    """Copy one system's trajectories + manifest from SOURCE_ROOT to
    TARGET_ROOT.  Returns (n_copied, n_skipped)."""
    src = SOURCE_ROOT / _subdir_name(compound, mp_id)
    dst = TARGET_ROOT / _subdir_name(compound, mp_id)
    if not src.is_dir():
        raise FileNotFoundError(f"Source not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_skipped = 0
    for f in sorted(src.iterdir()):
        if f.name.endswith(".tmp.npz"):
            continue  # ignore orphan tmp files
        target_file = dst / f.name
        if target_file.exists():
            n_skipped += 1
            continue
        shutil.copy2(f, target_file)
        n_copied += 1
    return n_copied, n_skipped


def _retrofit_npz(npz_path: Path) -> str:
    """Add shell_pair_* / shell_triplet_* arrays to a single .npz.

    Returns 'added', 'skipped', or 'failed: <reason>'.  Uses the same
    extraction pipeline as ``add_shell_tgt_to_big_dataset.py``.
    """
    # Lazy imports — heavy deps only needed when retrofitting.
    from ase.io import read as ase_read
    from tricor.relaxml.shell_target import extract_shell_target_arrays
    from tricor.shells import CoordinationShellTarget

    try:
        with np.load(npz_path) as npz:
            if "shell_pair_species" in npz.files:
                return "skipped"
            if "compound" not in npz.files or "mp_id" not in npz.files:
                return "failed: missing compound/mp_id key"
            compound = str(npz["compound"].item())
            mp_id = str(npz["mp_id"].item())
            payload = {
                k: npz[k] for k in npz.files
                if k not in (
                    "shell_pair_species", "shell_pair_features",
                    "shell_triplet_species", "shell_triplet_features",
                )
            }

        # Cache-friendly: caller passes one (compound, mp_id) at a time
        # through _retrofit_subdir, so we rebuild context once per subdir.
        global _CTX_CACHE
        key = (compound, mp_id)
        ctx = _CTX_CACHE.get(key)
        if ctx is None:
            cif_path = CIF_DIR / f"{mp_id}_{compound}.cif"
            if not cif_path.is_file():
                return f"failed: CIF not found at {cif_path}"
            ref = ase_read(str(cif_path), format="cif")
            target = CoordinationShellTarget.from_atoms(ref)
            ctx = extract_shell_target_arrays(target)
            _CTX_CACHE[key] = ctx
        payload.update(ctx)

        # Atomic rewrite.
        tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
        try:
            np.savez(tmp, **payload)
            shutil.move(str(tmp), str(npz_path))
        except Exception as e:
            if tmp.is_file():
                tmp.unlink()
            return f"failed: {type(e).__name__}: {e}"

        return "added"
    except Exception as e:
        return f"failed: {type(e).__name__}: {e}"


_CTX_CACHE: dict[tuple[str, str], dict] = {}


def _retrofit_subdir(compound: str, mp_id: str) -> tuple[int, int, int]:
    """Run retrofit over all .npz files in a curated subdir.
    Returns (added, skipped, failed)."""
    dst = TARGET_ROOT / _subdir_name(compound, mp_id)
    added = skipped = failed = 0
    failures: list[str] = []
    for npz in sorted(dst.glob("*.npz")):
        status = _retrofit_npz(npz)
        if status == "added":
            added += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            failures.append(f"{npz.name}: {status}")
    if failures:
        for f in failures[:5]:
            print(f"      [fail] {f}")
        if len(failures) > 5:
            print(f"      ... ({len(failures) - 5} more failures)")
    return added, skipped, failed


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"SOURCE_ROOT not found: {SOURCE_ROOT}")
    if RETROFIT and not CIF_DIR.is_dir():
        raise SystemExit(f"CIF_DIR not found: {CIF_DIR}")

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Curating {len(SYSTEMS)} systems")
    print(f"  Source: {SOURCE_ROOT}")
    print(f"  Target: {TARGET_ROOT}")
    print(f"  Retrofit shell_target: {RETROFIT}")
    print()

    t_start = time.perf_counter()
    total_copied = total_skipped = total_retrofitted = total_already_done = total_failed = 0

    for i, (compound, mp_id) in enumerate(SYSTEMS, start=1):
        print(f"[{i}/{len(SYSTEMS)}] {compound} / {mp_id}")
        n_copied, n_skipped = _copy_system(compound, mp_id)
        total_copied += n_copied
        total_skipped += n_skipped
        print(f"   copied {n_copied} files, skipped {n_skipped} (already in target)")

        if RETROFIT:
            n_add, n_done, n_fail = _retrofit_subdir(compound, mp_id)
            total_retrofitted += n_add
            total_already_done += n_done
            total_failed += n_fail
            print(
                f"   retrofit: added shell_target to {n_add}, "
                f"skipped (already had) {n_done}, failed {n_fail}"
            )

    dt = time.perf_counter() - t_start
    print()
    print(f"Done in {dt:.0f}s.")
    print(f"  Files copied:        {total_copied}")
    print(f"  Files already in dst: {total_skipped}")
    if RETROFIT:
        print(f"  Shell_target added:  {total_retrofitted}")
        print(f"  Already retrofitted: {total_already_done}")
        print(f"  Retrofit failed:     {total_failed}")
    print(f"\nEval set ready: {TARGET_ROOT}")


if __name__ == "__main__":
    main()
