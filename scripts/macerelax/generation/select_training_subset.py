"""select_training_subset.py — pick a stratified subset of the filtered corpus.

Reads corpus_filtered.csv (the output of filter_corpus.py), applies the
final empirically-revised exclusion rules, buckets CIFs by composition
class, randomly samples N per bucket, and writes cif_list.txt — the input
file that generate_mace_trajectories.py reads via its CIF_LIST_FILE
config.

Rules applied
-------------
HARD DROP (in addition to filter_corpus.py's drops):
  - any CIF flagged "actinide"        (~430 CIFs, includes Pu)
  - any CIF flagged "rare_radioactive" (~38 CIFs, mostly overlap with actinide)
KEEP (empirically validated by pilots):
  - lanthanide-flagged CIFs (8/8 healthy in test_lanthanide_quality.py)
  - U-corrected metal oxides (no spurious bond elongation in pilot)

Composition class buckets (mutually exclusive, priority top→bottom)
-------------------------------------------------------------------
  1. lanthanide   — contains any Ln (Z=57-71)
  2. tm_4d5d      — contains 4d/5d TM (Y-Cd, Hf-Hg) but no Ln
  3. tm_3d        — contains 3d TM (Sc-Zn) but no Ln, no 4d/5d TM
  4. hydride      — H atom fraction > 0.3 but no TM, no Ln
  5. main_group   — everything else (s- and p-block only)

Targets (sum = TARGET_TOTAL)
---------------------------
  See COMPOSITION_BUCKETS in the CONFIG block below.  Defaults sum to
  1,500 — the recommended corpus size for ~9,000-trajectory expansion.

If a bucket has fewer CIFs than its target, the script takes everything
available and reports the shortfall.

Output
------
  OUTPUT_LIST  — text file, one CIF filename per line (relative paths).
                  Format matches generate_mace_trajectories.py's
                  CIF_LIST_FILE expectation.
  OUTPUT_REPORT — text file with the per-bucket breakdown and the random
                  seed used for the sample (for reproducibility).

Run with:
    /home/ehrdt/miniforge3/envs/mace/bin/python \\
        scripts/macerelax/generation/select_training_subset.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# --- I/O ---
CORPUS_CSV    = Path("/pscratch/sd/e/ehrdt/tricor/cifs_mp_exp_1e100meV/corpus_filtered.csv")
OUTPUT_LIST   = Path("/pscratch/sd/e/ehrdt/tricor/cifs_mp_exp_1e100meV/training_subset.txt")
OUTPUT_REPORT = Path("/pscratch/sd/e/ehrdt/tricor/cifs_mp_exp_1e100meV/training_subset_report.txt")

# Existing-training-corpus dirs — any CIF whose filename OR mp_id appears
# in these directories will be excluded from the new subset (we don't want
# to re-generate trajectories for systems already in the training set).
# Empty list to disable.
EXCLUDE_DIRS = [
    Path("/pscratch/sd/e/ehrdt/tricor/cifs_mp_cnos_le100meV_training"),
]

# --- Sampling ---
SAMPLE_SEED = 42

# (bucket_name, target_count) — buckets are evaluated top-down; first match
# wins.  Targets sum to TARGET_TOTAL.
COMPOSITION_BUCKETS = [
    ("lanthanide",  300),
    ("tm_4d5d",     250),
    ("tm_3d",       400),
    ("hydride",      50),
    ("main_group",  500),
]

# --- Hard drops on top of filter_corpus.py's output ---
DROP_FLAGS = {"actinide", "rare_radioactive"}

# --- Hydride heuristic ---
# A CIF goes in the "hydride" bucket only if it's H-rich AND doesn't fall
# into one of the metallic / Ln buckets above (i.e., this is the residual
# main-group hydride pool).  Matches the h_rich flag set by filter_corpus.py.
H_RICH_FLAG = "h_rich"


# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import csv
import re
from collections import Counter
import numpy as np

_MP_ID_RE = re.compile(r"^(mp-\d+)_")


def existing_cif_basenames_and_mpids(dirs: list[Path]) -> tuple[set[str], set[str]]:
    """Walk each directory in dirs, collect all *.cif basenames + mp-ids.

    Returns (basenames, mp_ids).  Used to prevent re-selecting CIFs that
    are already in the existing training corpus.
    """
    basenames: set[str] = set()
    mp_ids:    set[str] = set()
    for d in dirs:
        if not d.is_dir():
            print(f"[warn] EXCLUDE_DIRS entry not found, skipping: {d}")
            continue
        for p in d.glob("*.cif"):
            basenames.add(p.name)
            m = _MP_ID_RE.match(p.stem)
            if m:
                mp_ids.add(m.group(1))
    return basenames, mp_ids


# ─────────────────────────────────────────────────────────────────────────────
# Composition class definitions
# ─────────────────────────────────────────────────────────────────────────────

LANTHANIDES = {  # Z=57-71
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}
TM_4D5D = {  # 4d (Y-Cd) + 5d (Hf-Hg)
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
}
TM_3D = {  # 3d (Sc-Zn)
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
}


def classify_composition(elements: set[str], flags: set[str]) -> str:
    """Return one of: lanthanide / tm_4d5d / tm_3d / hydride / main_group.

    Buckets are mutually exclusive — top-down priority order matches the
    COMPOSITION_BUCKETS list in CONFIG (lanthanide wins over any TM, TM
    wins over hydride/main-group).  An "other" return is reserved for the
    drop set; this function never returns it.
    """
    if elements & LANTHANIDES:
        return "lanthanide"
    if elements & TM_4D5D:
        return "tm_4d5d"
    if elements & TM_3D:
        return "tm_3d"
    if H_RICH_FLAG in flags:
        return "hydride"
    return "main_group"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CORPUS_CSV.is_file():
        raise SystemExit(f"[abort] CORPUS_CSV={CORPUS_CSV} not found "
                          f"— run filter_corpus.py first")

    with open(CORPUS_CSV) as f:
        rows = list(csv.DictReader(f))

    print(f"[subset] loaded {len(rows)} rows from {CORPUS_CSV}")

    # ── Collect existing-training CIFs (by basename + mp_id) ──────────────
    excluded_basenames, excluded_mp_ids = existing_cif_basenames_and_mpids(EXCLUDE_DIRS)
    if EXCLUDE_DIRS:
        print(f"[subset] exclusion sets from EXCLUDE_DIRS:")
        for d in EXCLUDE_DIRS:
            print(f"  {d}")
        print(f"  → {len(excluded_basenames)} basenames, "
              f"{len(excluded_mp_ids)} mp-ids")

    # ── Apply hard drops + bucket assignment ─────────────────────────────
    buckets: dict[str, list[dict]] = {name: [] for name, _ in COMPOSITION_BUCKETS}
    drop_count: Counter = Counter()

    for r in rows:
        if r["kept"].lower() != "true":
            # Already dropped by filter_corpus.py (composition filter)
            drop_count["filter_corpus.py"] += 1
            continue
        flags = set(f.strip() for f in r["flags"].split(";") if f.strip())
        if flags & DROP_FLAGS:
            drop_count["hard_drop"] += 1
            continue
        # Skip if already in the existing training corpus.
        if r["cif_filename"] in excluded_basenames:
            drop_count["already_in_training_basename"] += 1
            continue
        if r["mp_id"] and r["mp_id"] in excluded_mp_ids:
            drop_count["already_in_training_mp_id"] += 1
            continue
        elements = set(e.strip() for e in r["elements_present"].split(";")
                        if e.strip())
        bucket = classify_composition(elements, flags)
        buckets[bucket].append(r)

    print(f"\n[subset] available pool by bucket:")
    for name, n in COMPOSITION_BUCKETS:
        avail = len(buckets[name])
        print(f"  {name:12} target={n:>4}  available={avail:>4}")

    # ── Random sample per bucket ─────────────────────────────────────────
    rng = np.random.default_rng(SAMPLE_SEED)
    picked: list[dict] = []
    actual_per_bucket: dict[str, int] = {}
    shortfalls: dict[str, int] = {}

    for name, target in COMPOSITION_BUCKETS:
        pool = buckets[name]
        if len(pool) <= target:
            chosen = pool                # take everything
            shortfalls[name] = target - len(pool)
        else:
            idx = rng.choice(len(pool), size=target, replace=False)
            chosen = [pool[int(i)] for i in idx]
        actual_per_bucket[name] = len(chosen)
        picked.extend(chosen)

    # ── Write outputs ─────────────────────────────────────────────────────
    OUTPUT_LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LIST, "w") as f:
        for r in picked:
            f.write(r["cif_filename"] + "\n")

    with open(OUTPUT_REPORT, "w") as f:
        f.write("training_subset_report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"source CSV       : {CORPUS_CSV}\n")
        f.write(f"output list      : {OUTPUT_LIST}\n")
        f.write(f"random seed      : {SAMPLE_SEED}\n")
        f.write(f"drop flags       : {sorted(DROP_FLAGS)}\n\n")
        f.write(f"total rows in CSV    : {len(rows)}\n")
        f.write(f"already-dropped      : {drop_count['filter_corpus.py']}\n")
        f.write(f"hard-dropped here    : {drop_count['hard_drop']}\n")
        f.write(f"in existing training : "
                f"{drop_count.get('already_in_training_basename', 0)} basename + "
                f"{drop_count.get('already_in_training_mp_id', 0)} mp_id\n")
        f.write(f"available pool       : {sum(len(b) for b in buckets.values())}\n\n")
        f.write(f"per-bucket breakdown:\n")
        for name, target in COMPOSITION_BUCKETS:
            avail   = len(buckets[name])
            actual  = actual_per_bucket[name]
            short   = shortfalls.get(name, 0)
            short_note = f"  SHORTFALL={short}" if short else ""
            f.write(f"  {name:12} target={target:>4}  avail={avail:>4}  "
                    f"sampled={actual:>4}{short_note}\n")
        f.write(f"\nTOTAL SAMPLED    : {len(picked)}\n")

    # ── Console summary ──────────────────────────────────────────────────
    print(f"\n[subset] sampling complete (seed={SAMPLE_SEED}):")
    for name, target in COMPOSITION_BUCKETS:
        actual = actual_per_bucket[name]
        short  = shortfalls.get(name, 0)
        marker = f"  (-{short})" if short else ""
        print(f"  {name:12} {actual:>4}/{target:<4}{marker}")
    print(f"  {'TOTAL':12} {len(picked):>4}")
    print(f"\n[subset] wrote {OUTPUT_LIST}")
    print(f"[subset] wrote {OUTPUT_REPORT}")
    print(f"\nNext step: point generate_mace_trajectories.py's CIF_LIST_FILE at:")
    print(f"  {OUTPUT_LIST}")


if __name__ == "__main__":
    main()
