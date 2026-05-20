"""Filter the existing CIF library to a sibling dir of symlinks for CIFs
with e_above_hull <= MAX_E_ABOVE_HULL.

Reads every ``mp-XXX_Formula.cif`` filename in SRC_DIR, queries MP for
each material's energy_above_hull (one bulk call, chunked at
``CHUNK_SIZE``), and symlinks the passing CIFs into DST_DIR.  Existing
symlinks in DST_DIR are cleared first so re-runs reflect the current
cap without leaving stale links.

Implements §16.2 of the relaxml scope decision: existing CIF library
populated under the old 500 meV cap is reused; downstream generators
just point CIF_DIR at the new in-scope directory.

Usage:
    python scripts/relaxml/filter_cifs_by_ehull.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────
SRC_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos")
DST_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV")
MAX_E_ABOVE_HULL = 0.1  # eV/atom
MP_API_KEY_FILE = Path("/home/ehrdt/materials_project_api.txt")
CHUNK_SIZE = 1000  # mp-ids per MPRester call
EHULL_CACHE_FILE = DST_DIR.parent / "cifs_mp_cnos_ehull.json"
# ──────────────────────────────────────────────────────────────────────


def parse_mp_id(cif_path: Path) -> str:
    # filename: "mp-1234_Formula.cif" -> "mp-1234"
    return cif_path.stem.split("_", 1)[0]


def query_ehull(mp_ids: list[str]) -> dict[str, float]:
    """Bulk-query MP for energy_above_hull, chunked at CHUNK_SIZE."""
    from mp_api.client import MPRester
    api_key = MP_API_KEY_FILE.read_text().strip()
    out: dict[str, float] = {}
    with MPRester(api_key) as mpr:
        for i in range(0, len(mp_ids), CHUNK_SIZE):
            chunk = mp_ids[i : i + CHUNK_SIZE]
            docs = mpr.materials.summary.search(
                material_ids=chunk,
                fields=["material_id", "energy_above_hull"],
            )
            for d in docs:
                mid = str(d.material_id)
                e = d.energy_above_hull
                if e is None:
                    continue
                out[mid] = float(e)
            print(
                f"  queried {min(i + CHUNK_SIZE, len(mp_ids)):>5} / {len(mp_ids)} "
                f"  (this chunk returned {len(docs)} docs, cumulative {len(out)})",
                flush=True,
            )
    return out


def load_or_query_ehull(cif_files: list[Path]) -> dict[str, float]:
    """Return mp_id -> e_above_hull. Reuses cache when present."""
    file_ids = [parse_mp_id(p) for p in cif_files]
    if EHULL_CACHE_FILE.is_file():
        cache = json.loads(EHULL_CACHE_FILE.read_text())
        missing = sorted(set(file_ids) - set(cache))
        if missing:
            print(f"Cache hit: {len(cache)} ids cached, {len(missing)} new to query.")
            new = query_ehull(missing)
            cache.update(new)
            EHULL_CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
        else:
            print(f"Cache hit: all {len(cache)} ids cached.")
        return cache

    print(f"No cache at {EHULL_CACHE_FILE}; querying MP for {len(file_ids)} ids...")
    ehull = query_ehull(file_ids)
    EHULL_CACHE_FILE.write_text(json.dumps(ehull, indent=2, sort_keys=True) + "\n")
    return ehull


def main() -> None:
    if not SRC_DIR.is_dir():
        print(f"SRC_DIR missing: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    cif_files = sorted(SRC_DIR.glob("*.cif"))
    print(f"Scanning {SRC_DIR}: found {len(cif_files)} CIFs.")

    ehull = load_or_query_ehull(cif_files)
    print(
        f"\ne_above_hull resolved for {len(ehull)} / {len(cif_files)} files."
    )

    in_scope: list[Path] = []
    no_data: list[Path] = []
    for p in cif_files:
        mid = parse_mp_id(p)
        e = ehull.get(mid)
        if e is None:
            no_data.append(p)
        elif e <= MAX_E_ABOVE_HULL:
            in_scope.append(p)

    print(
        f"\nFilter @ e_above_hull <= {MAX_E_ABOVE_HULL} eV/atom:\n"
        f"  in scope    : {len(in_scope)}\n"
        f"  out of scope: {len(cif_files) - len(in_scope) - len(no_data)}\n"
        f"  no MP data  : {len(no_data)}"
    )

    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Clear stale symlinks (so re-runs at tighter caps don't leave links
    # to now-out-of-scope CIFs).  Refuse to touch non-symlink entries to
    # avoid trashing anything that ended up here by mistake.
    cleared = 0
    for existing in DST_DIR.iterdir():
        if existing.is_symlink():
            existing.unlink()
            cleared += 1
        else:
            print(
                f"WARN: refusing to remove non-symlink in DST_DIR: {existing}",
                file=sys.stderr,
            )
    if cleared:
        print(f"Cleared {cleared} stale symlinks in {DST_DIR}.")

    # Link the in-scope CIFs.  Also link hull_picks.json if present so
    # callers using _resolve_cif's cached hull picks see consistent state
    # — but only if the cached pick is still in scope.
    for src in in_scope:
        (DST_DIR / src.name).symlink_to(src.resolve())

    hull_picks_src = SRC_DIR / "hull_picks.json"
    if hull_picks_src.is_file():
        picks = json.loads(hull_picks_src.read_text())
        in_scope_names = {p.name for p in in_scope}
        kept = {k: v for k, v in picks.items() if v in in_scope_names}
        dropped = {k: v for k, v in picks.items() if v not in in_scope_names}
        (DST_DIR / "hull_picks.json").write_text(
            json.dumps(kept, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"\nhull_picks.json: kept {len(kept)} / {len(picks)} cached picks "
            f"(dropped {len(dropped)} out-of-scope picks)."
        )
        if dropped:
            print(
                "  dropped: "
                + ", ".join(f"{k}->{v}" for k, v in dropped.items())
            )

    print(f"\nDone. {len(in_scope)} symlinks written to {DST_DIR}/")


if __name__ == "__main__":
    main()
