"""Retrofit MACE+wall pilot trajectories with shell_target arrays.

Walks SOURCE_DIR recursively for .npz files (typically the whole
data/pilot_v1/{train,eval}/*/ tree), reads each file's system_id +
composition + mp_id directly from the NPZ (saved by the new generator),
resolves the corresponding reference CIF, computes the four shell_target
arrays once per system, and rewrites each .npz with those arrays
appended.

Adapted from scripts/relaxml/add_shell_tgt_to_npz.py — main differences:
  - Reads system identity (system_id, composition, mp_id) FROM the NPZ
    itself rather than from a hardcoded COMPOUND_PATTERNS/MP_IDS table.
    The new generator stores these as strings.  No dual-table sync needed.
  - Walks a nested {train,eval}/<system_id>/ layout via rglob.
  - shell_target context cache is keyed on system_id (so two SiO2
    polymorphs with different mp_id resolve to different shell_targets).

Idempotent: files that already have ``shell_pair_species`` are skipped
unless OVERWRITE=True.  Safe to run alongside or after the trajectory
generator.

Run: python scripts/macerelax/generation/add_shell_target_to_pilot.py
"""
from __future__ import annotations

# Limit BLAS threads before numpy import so this script doesn't grab
# every core on a shared box.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from tricor.macerelax.shell_target import extract_shell_target_arrays
from tricor.shells import CoordinationShellTarget

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

# Pilot dataset root; recurses through {train,eval}/<system_id>/*.npz.
SOURCE_DIR = Path("/home/ehrdt/tricor/mace/data/pilot_v1")

# CIF directory used by the pilot generator (must match what was used
# at trajectory generation time so the shell_target matches the polymorph).
CIF_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos")

# Set True to re-derive shell_target arrays even when a file already has
# them (use after a bug fix in the extractor).
OVERWRITE = False

# Resolve symlinks before rewriting so we modify the actual data file
# (not via a symlink chain).  Not relevant for the standard pilot layout
# but safe to leave on.
FOLLOW_SYMLINKS = True

# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _SystemContext:
    """Cached per-system shell_target arrays (built once, reused for
    every .npz of that system_id)."""
    system_id: str
    composition: str
    mp_id: str
    cif_path: Path
    arrays: dict[str, np.ndarray]


def _resolve_cif(composition: str, mp_id: str) -> Path:
    """Pin to {mp_id}_{composition}.cif exactly — the pilot generator
    always uses an explicit mp_id, so no glob/hull-pick fallback needed."""
    target = CIF_DIR / f"{mp_id}_{composition}.cif"
    if not target.is_file():
        raise FileNotFoundError(
            f"Pinned CIF not found: {target}. mp_id + composition from "
            f"the NPZ must point at a CIF that exists in CIF_DIR."
        )
    return target


def _build_context(system_id: str, composition: str,
                    mp_id: str) -> _SystemContext:
    cif_path = _resolve_cif(composition, mp_id)
    ref = ase_read(str(cif_path), format="cif")
    target = CoordinationShellTarget.from_atoms(ref)
    arrays = extract_shell_target_arrays(target)
    print(
        f"[{system_id:>22s}]  cif={cif_path.name}  "
        f"P={arrays['shell_pair_species'].shape[0]} "
        f"T={arrays['shell_triplet_species'].shape[0]}"
    )
    return _SystemContext(
        system_id=system_id, composition=composition, mp_id=mp_id,
        cif_path=cif_path, arrays=arrays,
    )


def _scalar_str(npz: np.lib.npyio.NpzFile, key: str) -> str:
    """Unwrap a 0-d string array (np.asarray("foo")) back to a Python str."""
    val = npz[key]
    return str(val.item()) if val.ndim == 0 else str(val)


def _system_keys_of(npz_path: Path,
                    npz: np.lib.npyio.NpzFile) -> tuple[str, str, str]:
    """Pull (system_id, composition, mp_id) from the NPZ. The new pilot
    generator always writes these as 0-d string arrays."""
    missing = [k for k in ("system_id", "composition", "mp_id")
               if k not in npz.files]
    # `composition` isn't in the generator's np.savez call as a separate
    # key — it's only in the manifest row. We always derive it the same
    # way the generator's SystemSpec did: composition is encoded in the
    # filename via the CIF lookup. Use mp_id to disambiguate.
    if "system_id" not in npz.files or "mp_id" not in npz.files:
        raise KeyError(
            f"{npz_path.name}: missing system_id/mp_id metadata. "
            f"This script expects pilot-v1-format NPZs."
        )
    system_id = _scalar_str(npz, "system_id")
    mp_id = _scalar_str(npz, "mp_id")
    # composition: the new generator does NOT save composition into the
    # NPZ (only into the manifest). Recover from the CIF lookup by
    # checking which {mp_id}_*.cif exists in CIF_DIR.
    matches = list(CIF_DIR.glob(f"{mp_id}_*.cif"))
    if not matches:
        raise FileNotFoundError(
            f"{npz_path.name}: no CIF in {CIF_DIR} matches "
            f"{mp_id}_*.cif (system_id={system_id!r})."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{npz_path.name}: multiple CIFs match {mp_id}_*.cif: "
            f"{[m.name for m in matches]}.  Ambiguous mp_id."
        )
    composition = matches[0].stem.split("_", 1)[1]
    return system_id, composition, mp_id


def _atomic_rewrite(npz_path: Path, new_payload: dict) -> None:
    """Write to a temp file in the same dir, then rename over the
    original. Avoids leaving a half-written .npz if the process dies."""
    target = npz_path.resolve() if FOLLOW_SYMLINKS else npz_path
    tmp = target.with_name(target.stem + ".tmp.npz")
    try:
        np.savez(tmp, **new_payload)
        shutil.move(str(tmp), str(target))
    except Exception:
        if tmp.is_file():
            tmp.unlink()
        raise


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"SOURCE_DIR not found: {SOURCE_DIR}")
    if not CIF_DIR.is_dir():
        raise SystemExit(f"CIF_DIR not found: {CIF_DIR}")

    npz_files = sorted(SOURCE_DIR.rglob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files under {SOURCE_DIR}")
    print(f"Scanning {len(npz_files)} .npz files under {SOURCE_DIR}\n")

    # system_id → _SystemContext cache (built lazily on first NPZ of each system)
    contexts: dict[str, _SystemContext] = {}
    n_added = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.perf_counter()

    for path in npz_files:
        try:
            with np.load(path) as npz:
                if (not OVERWRITE) and ("shell_pair_species" in npz.files):
                    n_skipped += 1
                    continue
                system_id, composition, mp_id = _system_keys_of(path, npz)
                if system_id not in contexts:
                    contexts[system_id] = _build_context(
                        system_id, composition, mp_id,
                    )
                ctx = contexts[system_id]
                # Read all existing fields, drop any prior shell_target
                # arrays (in case OVERWRITE), then add the new ones.
                payload = {
                    k: npz[k] for k in npz.files
                    if k not in (
                        "shell_pair_species", "shell_pair_features",
                        "shell_triplet_species", "shell_triplet_features",
                    )
                }
            payload.update(ctx.arrays)
            _atomic_rewrite(path, payload)
            n_added += 1
            if n_added % 50 == 0:
                print(f"  ...{n_added} files updated")
        except Exception as exc:
            print(f"  [fail] {path.name}: {type(exc).__name__}: {exc}")
            n_failed += 1

    dt = time.perf_counter() - t0
    print()
    print(f"Done in {dt:.1f}s.  added={n_added}  skipped={n_skipped}  "
          f"failed={n_failed}  systems={sorted(contexts)}")


if __name__ == "__main__":
    main()
