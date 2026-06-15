"""filter_corpus.py — apply MACE-MPA corpus-filtering rules.

Reads every CIF under CIF_DIR, applies the composition-based filter rules
documented in /home/ehrdt/tricor/MACE_CORPUS_FILTERING.md, and writes
corpus_filtered.csv with per-CIF kept/dropped status, drop reason, and
soft-flag annotations.

Rule recap
----------
HARD EXCLUDE if:
  1. Composition contains any element NOT in MACE-MPA-0's 88-element set
     (= Z=1-94 minus 6 noble gases He, Ne, Ar, Kr, Xe, Rn).
  2. Composition matches an organic-inorganic hybrid crystal heuristic
     (organic cation in a metal-halide or metal-oxide framework, e.g.
     methylammonium / formamidinium lead halide perovskites).

SOFT FLAG (include but tag in 'flags' column):
  - contains F                     → "fluoride"
  - contains Pu                    → "high_error_element"
  - H atom fraction > 0.3          → "h_rich"
  - contains lanthanide (Z 57-71)  → "lanthanide"
  - contains actinide (Z 89-94)    → "actinide"
  - contains Tc/Pm/Po/At/Fr/Ra/Ac  → "rare_radioactive"
  - contains Mn or Cu plus O       → "jahn_teller" (approximate)

Output
------
OUTPUT_CSV with one row per input CIF, columns:
  mp_id, formula, cif_filename, kept, drop_reason, flags,
  elements_present, n_atoms

The summary block printed at the end shows drop counts by reason and soft
flag distributions across the kept set.

Run with:
    /home/ehrdt/miniforge3/envs/mace/bin/python \\
        scripts/macerelax/generation/filter_corpus.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

CIF_DIR     = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV_training")
OUTPUT_CSV  = Path("/home/ehrdt/tricor/mace/data/corpus_filtered.csv")

# H atom fraction strictly greater than this triggers the "h_rich" flag.
H_RICH_THRESHOLD = 0.3

# Print a progress line every N CIFs.
PROGRESS_EVERY = 500

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import csv
import re
import sys
import traceback
from collections import Counter

from ase.io import read as ase_read
from ase.data import chemical_symbols


# ─────────────────────────────────────────────────────────────────────────────
# Element sets (derived from MACE_CORPUS_FILTERING.md §Decision 1)
# ─────────────────────────────────────────────────────────────────────────────

NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}

# Z=1 through Z=94 (Pu), minus the 6 noble gases above.
MACE_MPA_88 = {chemical_symbols[z] for z in range(1, 95)} - NOBLE_GASES
assert len(MACE_MPA_88) == 88, f"expected 88 elements, got {len(MACE_MPA_88)}"

# Soft-flag element categories.
LANTHANIDES      = {chemical_symbols[z] for z in range(57, 72)}   # La-Lu
ACTINIDES        = {chemical_symbols[z] for z in range(89, 95)}   # Ac-Pu
RARE_RADIOACTIVE = {"Tc", "Pm", "Po", "At", "Fr", "Ra", "Ac"}
JT_METALS        = {"Mn", "Cu"}   # approximate flag for Jahn-Teller-active oxides

# Hybrid-perovskite cation heuristic: explicit known cation strings in the
# formula plus a ratio-based fallback.
HYBRID_CATION_PATTERNS = re.compile(
    # Methylammonium fragments / variants and a few common others.
    r"CH3NH3|CH6N|CH5N2|C2H7N|C4H12N|HC\(NH2\)2",
    re.IGNORECASE,
)

# Metal-halide / metal-oxide host elements typical of hybrid perovskites.
HYBRID_METAL_HOSTS  = {"Pb", "Sn", "Bi", "Sb", "Ge"}
HYBRID_HALIDE_HOSTS = {"I", "Br", "Cl"}

# Filename → (mp_id, formula) parser (matches generate_mace_trajectories.py).
_MP_ID_RE = re.compile(r"^(mp-\d+)_(.+)$")


# ─────────────────────────────────────────────────────────────────────────────
# Per-CIF analysis
# ─────────────────────────────────────────────────────────────────────────────

def parse_cif_name(cif_path: Path) -> tuple[str, str]:
    """Extract (mp_id, formula) from a filename like 'mp-19770_Fe2O3.cif'.

    Falls back to ('', stem) if the pattern doesn't match — caller can still
    process by re-deriving the formula from the CIF body.
    """
    m = _MP_ID_RE.match(cif_path.stem)
    if not m:
        return "", cif_path.stem
    return m.group(1), m.group(2)


def composition_from_cif(cif_path: Path) -> tuple[Counter, str]:
    """Read a CIF, return (element counter, reduced formula)."""
    atoms = ase_read(str(cif_path), format="cif")
    symbols = atoms.get_chemical_symbols()
    return Counter(symbols), atoms.get_chemical_formula()


def check_hard_excludes(elements: Counter, formula: str) -> str | None:
    """Return a drop-reason string if the CIF should be hard-excluded,
    otherwise None.  Drop reasons have the shape '<category>:<detail>' so
    the summary can aggregate by category."""

    # Rule 1: 88-element coverage.
    unsupported = elements.keys() - MACE_MPA_88
    if unsupported:
        return f"unsupported_element:{','.join(sorted(unsupported))}"

    # Rule 2: organic-inorganic hybrid crystal.
    has_C = "C" in elements
    has_H = "H" in elements
    has_N = "N" in elements

    if has_C and has_H and has_N:
        # Heuristic A: explicit cation string match.
        if HYBRID_CATION_PATTERNS.search(formula):
            return "hybrid_cation:formula_regex"

        # Heuristic B: H-rich C+N content alongside a metal-halide /
        # metal-oxide host framework.  Organic cations have H/(C+N) ≳ 2.5;
        # an inorganic carbide/nitride won't.
        h     = elements["H"]
        cn    = elements["C"] + elements["N"]
        if cn > 0 and h / cn >= 2.5:
            host_metal  = bool(elements.keys() & HYBRID_METAL_HOSTS)
            has_halide  = bool(elements.keys() & HYBRID_HALIDE_HOSTS)
            has_oxide   = "O" in elements
            if host_metal and (has_halide or has_oxide):
                return "hybrid_cation:H_rich_with_metal_framework"

    return None


def compute_soft_flags(elements: Counter, n_atoms: int) -> list[str]:
    """Return a list of soft-flag strings.  All CIFs that pass hard excludes
    are kept; flags are non-exclusive annotations.
    """
    flags: list[str] = []

    if "F" in elements:
        flags.append("fluoride")
    if "Pu" in elements:
        flags.append("high_error_element")
    if n_atoms > 0 and elements.get("H", 0) / n_atoms > H_RICH_THRESHOLD:
        flags.append("h_rich")
    if elements.keys() & LANTHANIDES:
        flags.append("lanthanide")
    if elements.keys() & ACTINIDES:
        flags.append("actinide")
    if elements.keys() & RARE_RADIOACTIVE:
        flags.append("rare_radioactive")
    if (elements.keys() & JT_METALS) and "O" in elements:
        flags.append("jahn_teller")

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cif_paths = sorted(CIF_DIR.glob("*.cif"))
    if not cif_paths:
        raise SystemExit(f"[abort] no CIFs found in {CIF_DIR}")

    print(f"[filter] CIF_DIR    : {CIF_DIR}")
    print(f"[filter] OUTPUT_CSV : {OUTPUT_CSV}")
    print(f"[filter] processing {len(cif_paths)} CIFs")
    print()

    rows: list[dict] = []
    drop_counter: Counter = Counter()
    flag_counter: Counter = Counter()

    for i, path in enumerate(cif_paths, 1):
        if i % PROGRESS_EVERY == 0:
            print(f"  [{i}/{len(cif_paths)}] processed")

        mp_id, name_formula = parse_cif_name(path)

        try:
            elements, parsed_formula = composition_from_cif(path)
            n_atoms = sum(elements.values())
        except Exception as exc:
            # Capture the read error so the user can audit problematic files.
            err = f"read_error:{type(exc).__name__}"
            drop_counter["read_error"] += 1
            rows.append({
                "mp_id":            mp_id,
                "formula":          name_formula,
                "cif_filename":     path.name,
                "kept":             False,
                "drop_reason":      err,
                "flags":            "",
                "elements_present": "",
                "n_atoms":          0,
            })
            continue

        drop_reason = check_hard_excludes(elements, parsed_formula)

        if drop_reason is None:
            flags = compute_soft_flags(elements, n_atoms)
            for f in flags:
                flag_counter[f] += 1
            rows.append({
                "mp_id":            mp_id,
                "formula":          parsed_formula,
                "cif_filename":     path.name,
                "kept":             True,
                "drop_reason":      "",
                "flags":            ";".join(flags),
                "elements_present": ";".join(sorted(elements.keys())),
                "n_atoms":          n_atoms,
            })
        else:
            # Aggregate by category (text before the first colon) for the report.
            drop_counter[drop_reason.split(":", 1)[0]] += 1
            rows.append({
                "mp_id":            mp_id,
                "formula":          parsed_formula,
                "cif_filename":     path.name,
                "kept":             False,
                "drop_reason":      drop_reason,
                "flags":            "",
                "elements_present": ";".join(sorted(elements.keys())),
                "n_atoms":          n_atoms,
            })

    # ── Write CSV ──────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mp_id", "formula", "cif_filename", "kept", "drop_reason",
        "flags", "elements_present", "n_atoms",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary report ────────────────────────────────────────────────────
    n_total   = len(rows)
    n_kept    = sum(1 for r in rows if r["kept"])
    n_dropped = n_total - n_kept

    print()
    print("=" * 60)
    print(f"  Total CIFs    : {n_total}")
    print(f"  Kept          : {n_kept}  ({n_kept / n_total * 100:.1f}%)")
    print(f"  Dropped       : {n_dropped}  ({n_dropped / n_total * 100:.1f}%)")

    if drop_counter:
        print()
        print("  Drop categories:")
        for cat, count in sorted(drop_counter.items(), key=lambda x: -x[1]):
            print(f"    {cat:30} {count:>6}")

    if flag_counter:
        print()
        print("  Soft flags applied (in the kept set):")
        for flag, count in sorted(flag_counter.items(), key=lambda x: -x[1]):
            pct = count / n_kept * 100 if n_kept else 0.0
            print(f"    {flag:30} {count:>6}  ({pct:.1f}% of kept)")

    print()
    print(f"  Wrote {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
