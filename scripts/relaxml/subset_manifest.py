"""Subsample a merged trajectory manifest by compound.

Reads the merged manifest, keeps the first N rows per chosen compound,
symlinks the corresponding .npz files into a destination dir, writes a
subset manifest CSV.  Used to carve a fast iteration loop out of a large
multi-compound corpus without regenerating data.

Edit the CONFIG block, then run:
    python subset_manifest.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_MANIFEST =  "./data/si-n-trajectories/manifest.csv" # "./data/multicomp_trajectories_merged/manifest.csv"
DESTINATION = "./data/si-n-trajectories/300_subset"

# Compounds to include + how many trajectories to keep per compound.
# Selection is the first N rows in manifest order (deterministic; the
# generator writes rows by stratum × idx so this gives balanced regimes).
COMPOUNDS = ["Si", "SiC", "SiO2", "BN", "AlN"]
N_PER_COMPOUND = 300

# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    src_manifest = Path(SOURCE_MANIFEST).resolve()
    src_dir = src_manifest.parent
    dst_dir = Path(DESTINATION).resolve()

    if not src_manifest.is_file():
        raise SystemExit(f"Source manifest not found: {src_manifest}")

    with open(src_manifest, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "compound" not in (reader.fieldnames or []):
            raise SystemExit(
                f"{src_manifest} has no 'compound' column. Was it produced "
                f"by the multi-species generator?"
            )
        all_rows = list(reader)

    by_compound: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_compound[row["compound"]].append(row)

    chosen_rows: list[dict] = []
    for cname in COMPOUNDS:
        rows = by_compound.get(cname, [])
        if not rows:
            print(f"  [warn] no rows for compound {cname!r} in source manifest")
            continue
        n_take = min(N_PER_COMPOUND, len(rows))
        if len(rows) < N_PER_COMPOUND:
            print(f"  [warn] {cname}: only {len(rows)} rows available "
                  f"(< requested {N_PER_COMPOUND})")
        chosen_rows.extend(rows[:n_take])
        print(f"  {cname}: kept {n_take} of {len(rows)}")

    if not chosen_rows:
        raise SystemExit("No rows kept; check COMPOUNDS list.")

    dst_dir.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    n_missing = 0
    for row in chosen_rows:
        src_npz = src_dir / row["filename"]
        # Resolve through any existing symlinks so the subset dir contains
        # one-hop links directly to the original .npz files (no chains).
        if src_npz.is_symlink():
            target = src_npz.resolve()
        elif src_npz.is_file():
            target = src_npz
        else:
            print(f"  [warn] missing: {src_npz}")
            n_missing += 1
            continue
        link = dst_dir / row["filename"]
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
        n_linked += 1

    out_manifest = dst_dir / "manifest.csv"
    with open(out_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(chosen_rows[0].keys()))
        writer.writeheader()
        writer.writerows(chosen_rows)

    n_npz = len(list(dst_dir.glob("*.npz")))
    print()
    print(f"Subset manifest: {out_manifest}")
    print(f"  rows:       {len(chosen_rows)}")
    print(f"  .npz files: {n_npz}  (linked: {n_linked}, missing: {n_missing})")
    print(f"  compounds:  {sorted(set(r['compound'] for r in chosen_rows))}")


if __name__ == "__main__":
    main()
