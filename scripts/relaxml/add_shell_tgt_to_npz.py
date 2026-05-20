"""Retrofit existing .npz trajectories with shell_target arrays.

Walks SOURCE_DIR for .npz files, derives the shell_target from each
file's compound (parsed from the filename, or read from a top-level
``compound`` field if present), and rewrites the file with the four
shell_target arrays added to the schema.  No re-running of shell_relax —
the dynamics are unchanged, we're just adding conditioning input that
the model will see at training time.

Edit the CONFIG block below, then run:
    python add_shell_target_to_npz.py

Idempotent: files that already contain shell_pair_species are skipped
unless OVERWRITE=True.

KEEP IN SYNC with the COMPOUND_PRESETS in
``scripts/flowmatch_guided/generate_surrogate_trajectories.py``.
"""

from __future__ import annotations

# Limit BLAS threads before numpy import so this script doesn't grab
# every core on a shared box.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from tricor.relaxml.shell_target import extract_shell_target_arrays
from tricor.shells import CoordinationShellTarget

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

# Walk this directory recursively (or just at the top level) for .npz files.
# Use the merged dir to retrofit every compound at once, OR a single
# per-compound dir to scope the rewrite.
SOURCE_DIR =  "/wigeon/users/ehrdt/prod/relaxml_big_v1/" #"./data/multi_species_v2/Si3N4_trajectories_150"

CIF_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos")
HULL_PICKS_FILE = CIF_DIR / "hull_picks.json"

# Compound name → CIF glob pattern.  Mirrors COMPOUND_PRESETS in
# generate_surrogate_trajectories.py.
COMPOUND_PATTERNS: dict[str, str] = {
    "Si":     "*_Si.cif",
    "Ge":     "*_Ge.cif",
    "SiC":    "*_SiC.cif",
    "BN":     "*_BN.cif",
    "AlN":     "*_AlN.cif",
    "Si3N4":  "*_Si3N4.cif",
    "SiO2":   "*_SiO2.cif",
    "GeO2":   "*_GeO2.cif",
    "B2O3":   "*_B2O3.cif",
    "Al2O3":  "*_Al2O3.cif",
    "Ga2O3":  "*_Ga2O3.cif",
    "TiO2":   "*_TiO2.cif",
    "As2S3":  "*_As2S3.cif",
}

# Per-compound mp-id pins.  When a compound has multiple polymorphs in MP
# (Al2O3 corundum vs other Al2O3 phases, Ga2O3 monoclinic-β vs α, TiO2
# anatase vs rutile vs brookite, ...), specify which one to use for
# shell_target extraction.  This MUST match the mp-id you used when
# generating the trajectories — otherwise the shell_target the model is
# conditioned on at training time won't match the structure it's
# relaxing.
#
# Empty/missing entries fall back to the hull pick (lowest e_above_hull
# from MP, cached in hull_picks.json), which is the correct default for
# compounds that only have one stable polymorph in the dataset.
#
# Mirrors CompoundSpec.mp_id in generate_surrogate_trajectories_multicomp.py.
COMPOUND_MP_IDS: dict[str, str] = {
    # "Al2O3": "mp-1143",   # corundum (only fill when needed)
    "Ga2O3": "mp-886",    # β-Ga2O3 monoclinic — confirm with `ls *_Ga2O3.cif`
    "TiO2":  "mp-390",    # anatase — confirm with `ls *_TiO2.cif`
}

# Set True to re-derive shell_target arrays even when a file already has
# them (use after fixing a bug in the extractor).
OVERWRITE = False

# Resolve symlinks before rewriting so we modify the actual data file
# (not via a symlink chain).  Recommended True when SOURCE_DIR is the
# merged/ symlink farm.
FOLLOW_SYMLINKS = True

# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _CompoundContext:
    """Cached per-compound shell_target arrays (built once, reused for
    every .npz of that compound)."""
    compound: str
    cif_path: Path
    arrays: dict[str, np.ndarray]


def _resolve_cif(compound: str) -> Path:
    """Resolve the CIF for ``compound``.

    Resolution order:
      1. ``COMPOUND_MP_IDS[compound]`` — pin to ``{mp_id}_{compound}.cif``
         exactly.  Use when the trajectories were generated with a
         specific mp-id pin (e.g. Ga2O3 β-phase, TiO2 anatase).  The
         shell_target MUST come from the same polymorph the trajectories
         were generated from, otherwise training/inference is misaligned.
      2. Single CIF match for ``COMPOUND_PATTERNS[compound]`` glob.
      3. Multiple matches: hull pick from ``hull_picks.json`` cache.
    """
    # 1. Explicit mp-id pin: skip globbing entirely, use the exact filename.
    pinned_mp = COMPOUND_MP_IDS.get(compound)
    if pinned_mp is not None:
        target = CIF_DIR / f"{pinned_mp}_{compound}.cif"
        if not target.is_file():
            raise FileNotFoundError(
                f"Pinned CIF not found: {target}.  Verify mp_id={pinned_mp!r} "
                f"is correct for compound {compound!r} (check "
                f"`ls {CIF_DIR}/*_{compound}.cif`).  COMPOUND_MP_IDS "
                f"must agree with the mp-id used at generation time."
            )
        return target

    # 2. Glob fallback: pattern lookup + hull-pick if ambiguous.
    pattern = COMPOUND_PATTERNS.get(compound)
    if pattern is None:
        raise KeyError(
            f"No CIF pattern for compound {compound!r}.  Add it to "
            f"COMPOUND_PATTERNS, keeping it in sync with the generation "
            f"script."
        )
    matches = sorted(CIF_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No CIF in {CIF_DIR} matching {pattern!r} for {compound!r}."
        )
    if len(matches) == 1:
        return matches[0]
    cache: dict[str, str] = {}
    if HULL_PICKS_FILE.is_file():
        cache = json.loads(HULL_PICKS_FILE.read_text())
    cached_name = cache.get(compound)
    if cached_name is None:
        listed = "\n    ".join(m.name for m in matches)
        raise ValueError(
            f"{len(matches)} CIFs match {pattern!r} for {compound!r} and "
            f"there's no entry in {HULL_PICKS_FILE.name}.  Either run the "
            f"generator once to populate the cache, or hand-edit the file, "
            f"or pin via COMPOUND_MP_IDS in this script:"
            f"\n    {listed}"
        )
    chosen = CIF_DIR / cached_name
    if not chosen.is_file():
        raise FileNotFoundError(
            f"hull_picks chose {cached_name} for {compound!r} but the file "
            f"is missing from {CIF_DIR}."
        )
    return chosen


def _build_context(compound: str) -> _CompoundContext:
    cif_path = _resolve_cif(compound)
    ref = ase_read(str(cif_path), format="cif")
    target = CoordinationShellTarget.from_atoms(ref)
    arrays = extract_shell_target_arrays(target)
    print(
        f"[{compound:>6s}] {cif_path.name}  "
        f"P={arrays['shell_pair_species'].shape[0]} "
        f"T={arrays['shell_triplet_species'].shape[0]}"
    )
    return _CompoundContext(compound=compound, cif_path=cif_path, arrays=arrays)


def _compound_of(npz_path: Path, npz: np.lib.npyio.NpzFile) -> str:
    """Prefer the in-file 'compound' field; fall back to filename prefix."""
    if "compound" in npz.files:
        return str(npz["compound"].item())
    # Filenames look like "<compound>_<regime>_cell###_idx#####_seed#########.npz"
    return npz_path.stem.split("_", 1)[0]


def _atomic_rewrite(npz_path: Path, new_payload: dict) -> None:
    """Write to a temp file in the same dir, then rename over the
    original.  Avoids leaving a half-written .npz if the process dies.

    The temp path already ends in ``.npz`` so ``np.savez`` doesn't
    silently append another ``.npz`` extension.
    """
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
    src = Path(SOURCE_DIR).resolve()
    if not src.is_dir():
        raise SystemExit(f"SOURCE_DIR not found: {src}")
    if not CIF_DIR.is_dir():
        raise SystemExit(f"CIF_DIR not found: {CIF_DIR}")

    npz_files = sorted(src.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files in {src}")
    print(f"Scanning {len(npz_files)} .npz files in {src}\n")

    contexts: dict[str, _CompoundContext] = {}
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
                compound = _compound_of(path, npz)
                if compound not in contexts:
                    contexts[compound] = _build_context(compound)
                ctx = contexts[compound]
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
        except Exception as e:
            print(f"  [fail] {path.name}: {type(e).__name__}: {e}")
            n_failed += 1

    dt = time.perf_counter() - t0
    print()
    print(f"Done in {dt:.1f}s.  added={n_added}  skipped={n_skipped}  "
          f"failed={n_failed}  compounds={sorted(contexts)}")


if __name__ == "__main__":
    main()
