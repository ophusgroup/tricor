"""Patch the merged big-dataset manifest so ``filename`` includes the
per-CIF subdir prefix.

``generate_big_dataset.py`` writes the merged manifest by concatenating
per-CIF manifest rows verbatim.  Those rows have bare ``filename``
values (no path) because each per-CIF manifest lives in the same
directory as its .npz files.  At the top level, the .npz files live in
``{compound}_{mp_id}_trajectories/`` subdirs — so the dataloader's
``data_root / filename`` lookup misses.

Fix: prepend ``{compound}_{mp_id}_trajectories/`` to each row's
``filename``.  Idempotent (skips rows already prefixed).  Writes a
``.bak`` of the original before rewriting in place.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

MANIFEST = Path("/wigeon/users/ehrdt/prod/relaxml_big_v1/manifest.csv")


def _expected_subdir(compound: str, mp_id: str) -> str:
    return f"{compound}_{mp_id}_trajectories"


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Manifest not found: {MANIFEST}")

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
        headers = list(rows[0].keys()) if rows else []

    if not rows:
        raise SystemExit("Manifest is empty")

    n_changed = 0
    n_already = 0
    n_missing_subdir = 0

    for row in rows:
        subdir = _expected_subdir(row["compound"], row["mp_id"])
        fname = row["filename"]
        if fname.startswith(subdir + "/"):
            n_already += 1
            continue
        if "/" in fname:
            n_missing_subdir += 1
            continue
        row["filename"] = f"{subdir}/{fname}"
        n_changed += 1

    if n_changed == 0:
        print(f"Nothing to do.  already={n_already}  weird={n_missing_subdir}")
        return

    # Back up before rewriting.
    bak = MANIFEST.with_suffix(".csv.bak")
    if not bak.is_file():
        shutil.copy2(MANIFEST, bak)
        print(f"Backup written: {bak}")

    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(MANIFEST)

    print(
        f"Patched.  changed={n_changed}  already_prefixed={n_already}  "
        f"unexpected_path={n_missing_subdir}"
    )

    # Quick verification: spot-check one row resolves to a real file.
    sample = rows[0]
    resolved = MANIFEST.parent / sample["filename"]
    print(f"\nSpot check: {resolved} exists={resolved.is_file()}")


if __name__ == "__main__":
    main()
