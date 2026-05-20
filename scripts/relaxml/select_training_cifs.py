"""Select a stratified subset of the in-scope CIF library for big-model
training data generation.

Reads ``SRC_DIR`` (typically ``/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV/``),
buckets every CIF by primary anion using the same priority ordering as
``generate_cnos.py`` (O > N > S > C, plus a monatomic bucket), and
selects:

  1. Every CIF in ``MUST_INCLUDE`` is force-included (eval-set continuity).
  2. Exactly ``RANDOM_PER_BUCKET`` CIFs sampled uniformly from each bucket,
     drawn from the pool of in-scope CIFs minus the must-includes.

Flat per-bucket allocation rather than proportional: gives every element
family the same training-data presence regardless of MP coverage.  Total
selection size = 5 * RANDOM_PER_BUCKET + len(MUST_INCLUDE) (modulo
bucket-size caps for small buckets).

Writes results in two forms:

  * A sibling symlink directory (``DST_DIR``) — generation scripts just
    point their CIF_DIR at this path.
  * A text file ``selected_cifs.txt`` (filenames, one per line) for
    record-keeping.

Implements §16.4 (rescoped per §13.11): produces the CIF subset for the
10k-trajectory breadth-heavy training corpus.

Usage:
    python scripts/relaxml/select_training_cifs.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────
SRC_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV")
DST_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV_training")

RANDOM_PER_BUCKET = 100  # uniform random draws from each anion bucket
SEED = 0

# Same ordering as generate_cnos.py: classify a binary by its highest-
# electronegativity anion.  Monatomics go in their own bucket regardless
# of element.
ANION_PRIORITY = ["O", "N", "S", "C"]
BUCKETS = ["monatomic"] + ANION_PRIORITY

# CIFs forced into the selection.  Keeps the existing eval set in scope.
# Filenames must match what's in SRC_DIR (mp-XXX_Formula.cif).
MUST_INCLUDE = [
    # Si-N cross-composition set (replicates the original si-n_phys
    # training corpus so the new big model is at least a superset).
    "mp-149_Si.cif",            # Si ground state
    "mp-1204356_SiC.cif",       # SiC (hull pick)
    "mp-7000_SiO2.cif",         # SiO2 quartz
    "mp-661_AlN.cif",           # AlN
    "mp-604884_BN.cif",         # BN
    "mp-988_Si3N4.cif",         # Si3N4 (was held-out; now in training too)
    # Existing oxide eval / cross-coord set.
    "mp-1143_Al2O3.cif",        # corundum
    "mp-886_Ga2O3.cif",         # β-Ga2O3
    "mp-390_TiO2.cif",          # anatase
    # Additional in-scope SiO2 polymorphs we already have trajectory
    # data for (per Part 6 of RELAXML_SESSION.txt).
    "mp-6930_SiO2.cif",         # coesite
    "mp-6945_SiO2.cif",         # α-cristobalite
    "mp-546794_SiO2.cif",       # β-cristobalite
    # Additional TiO2 / Al2O3 polymorphs worth seeding (within scope).
    "mp-2657_TiO2.cif",         # rutile
    "mp-1840_TiO2.cif",         # brookite
    # GeO2 / Ge for broader Group-IV oxide coverage.
    "mp-470_GeO2.cif",
    "mp-32_Ge.cif",
]
# ──────────────────────────────────────────────────────────────────────


_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)\d*")


def parse_formula(cif_path: Path) -> str:
    """Extract formula from ``mp-XXX_Formula.cif`` filename."""
    stem = cif_path.stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def bucket_for(cif_path: Path) -> str:
    formula = parse_formula(cif_path)
    elements = set(_ELEMENT_RE.findall(formula))
    if len(elements) == 1:
        return "monatomic"
    for el in ANION_PRIORITY:
        if el in elements:
            return el
    # Binaries with no C/N/O/S anion — shouldn't appear if SRC_DIR was
    # built by generate_cnos.py, but flag rather than silently dropping.
    return "other"


def allocate_flat(
    bucket_sizes: dict[str, int],
    per_bucket: int,
    fixed_consumed: dict[str, int],
) -> dict[str, int]:
    """Flat per-bucket allocation: ``per_bucket`` random draws from each
    non-empty bucket, capped by availability (size - fixed)."""
    return {
        b: min(per_bucket, max(0, size - fixed_consumed.get(b, 0)))
        for b, size in bucket_sizes.items()
    }


def main() -> None:
    if not SRC_DIR.is_dir():
        print(f"SRC_DIR missing: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    cif_files = sorted(SRC_DIR.glob("*.cif"))
    print(f"Scanning {SRC_DIR}: found {len(cif_files)} CIFs.")

    # Bucket every CIF.
    by_bucket: dict[str, list[Path]] = defaultdict(list)
    for p in cif_files:
        by_bucket[bucket_for(p)].append(p)
    for b in BUCKETS + ["other"]:
        by_bucket.setdefault(b, [])
    bucket_sizes = {b: len(v) for b, v in by_bucket.items()}
    print("\nBucket sizes (full in-scope library):")
    for b in BUCKETS + (["other"] if by_bucket["other"] else []):
        print(f"  {b:>11}: {bucket_sizes[b]:>4}")

    # Resolve must-include set, splitting by bucket.
    name_to_path = {p.name: p for p in cif_files}
    must_include_paths: list[Path] = []
    missing: list[str] = []
    for name in MUST_INCLUDE:
        if name in name_to_path:
            must_include_paths.append(name_to_path[name])
        else:
            missing.append(name)
    if missing:
        print(
            f"\nWARN: {len(missing)} must-include CIF(s) not in SRC_DIR:\n  "
            + "\n  ".join(missing)
        )

    fixed_consumed: dict[str, int] = defaultdict(int)
    for p in must_include_paths:
        fixed_consumed[bucket_for(p)] += 1
    fixed_consumed = dict(fixed_consumed)
    print(
        f"\nMust-include: {len(must_include_paths)} CIFs "
        f"(by bucket: {dict(fixed_consumed)})"
    )

    quota = allocate_flat(bucket_sizes, RANDOM_PER_BUCKET, fixed_consumed)
    print("\nAllocation plan (flat per-bucket + must-includes):")
    print(f"  {'bucket':>11}  {'size':>5}  {'fixed':>5}  {'random':>6}  {'total':>5}")
    for b in BUCKETS + (["other"] if by_bucket["other"] else []):
        size = bucket_sizes[b]
        fixed = fixed_consumed.get(b, 0)
        rnd = quota.get(b, 0)
        print(f"  {b:>11}  {size:>5}  {fixed:>5}  {rnd:>6}  {fixed + rnd:>5}")

    total_planned = sum(fixed_consumed.values()) + sum(quota.values())
    print(f"  {'TOTAL':>11}  {len(cif_files):>5}  "
          f"{sum(fixed_consumed.values()):>5}  "
          f"{sum(quota.values()):>6}  {total_planned:>5}")
    print(f"\nRANDOM_PER_BUCKET = {RANDOM_PER_BUCKET}; planned total = {total_planned}.")

    # Sample within each bucket, excluding already-fixed paths.
    rng = np.random.default_rng(SEED)
    fixed_set = set(must_include_paths)
    selected: list[Path] = list(must_include_paths)
    for b, q in quota.items():
        if q <= 0:
            continue
        pool = [p for p in by_bucket[b] if p not in fixed_set]
        if len(pool) <= q:
            selected.extend(pool)
            continue
        idx = rng.choice(len(pool), size=q, replace=False)
        selected.extend(pool[i] for i in idx)

    selected_unique: list[Path] = []
    seen: set[Path] = set()
    for p in selected:
        if p not in seen:
            seen.add(p)
            selected_unique.append(p)
    selected_unique.sort()

    print(f"\nFinal selection: {len(selected_unique)} CIFs.")

    # Write outputs.
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Clear stale symlinks; refuse to touch non-symlinks.
    cleared = 0
    for existing in DST_DIR.iterdir():
        if existing.is_symlink():
            existing.unlink()
            cleared += 1
        elif existing.name in {"selected_cifs.txt", "selection_report.json",
                                "hull_picks.json"}:
            existing.unlink()
            cleared += 1
        else:
            print(f"WARN: refusing to remove {existing}", file=sys.stderr)
    if cleared:
        print(f"Cleared {cleared} entries from {DST_DIR}.")

    for src in selected_unique:
        (DST_DIR / src.name).symlink_to(src.resolve())

    (DST_DIR / "selected_cifs.txt").write_text(
        "\n".join(p.name for p in selected_unique) + "\n"
    )

    # Carry the in-scope hull_picks.json forward, filtered to selection.
    hull_picks_src = SRC_DIR / "hull_picks.json"
    if hull_picks_src.is_file():
        picks = json.loads(hull_picks_src.read_text())
        selected_names = {p.name for p in selected_unique}
        kept = {k: v for k, v in picks.items() if v in selected_names}
        (DST_DIR / "hull_picks.json").write_text(
            json.dumps(kept, indent=2, sort_keys=True) + "\n"
        )
        dropped = {k: v for k, v in picks.items() if v not in selected_names}
        if dropped:
            print(
                f"hull_picks.json: kept {len(kept)} / {len(picks)} "
                f"(dropped {len(dropped)} not in selection)"
            )

    # Selection report for the manifest.
    report = {
        "src_dir": str(SRC_DIR),
        "dst_dir": str(DST_DIR),
        "random_per_bucket": RANDOM_PER_BUCKET,
        "seed": SEED,
        "bucket_sizes": bucket_sizes,
        "fixed_consumed": fixed_consumed,
        "quota": quota,
        "n_must_include_found": len(must_include_paths),
        "n_must_include_missing": len(missing),
        "missing_must_include": missing,
        "n_selected": len(selected_unique),
    }
    (DST_DIR / "selection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    print(f"\nDone. {len(selected_unique)} symlinks + manifests in {DST_DIR}/")


if __name__ == "__main__":
    main()
