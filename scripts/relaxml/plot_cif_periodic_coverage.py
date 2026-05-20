"""Render a periodic-table heatmap showing how many CIFs in a selection
directory contain each element.

Reads every ``*.cif`` in ``CIF_DIR`` (filename pattern
``mp-XXX_Formula.cif``), parses each formula into element symbols, and
counts CIFs containing each element.  Renders the result as a
periodic-table-layout heatmap with element symbols + counts.

Usage:
    # Use the configured CIF_DIR below.
    python scripts/relaxml/plot_cif_periodic_coverage.py

    # Or override at the CLI with any other CIF dir.
    python scripts/relaxml/plot_cif_periodic_coverage.py /path/to/cifs

The output PNG is written to ``<cif_dir>/periodic_table_coverage.png``.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from matplotlib.colors import LogNorm


# ── CONFIG ────────────────────────────────────────────────────────────
CIF_DIR = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos_le100meV_training")
OUT_PATH = CIF_DIR / "periodic_table_coverage.png"
# ──────────────────────────────────────────────────────────────────────


# Standard periodic-table layout, (row, col) per element.  Rows 1-7 are
# the main grid; rows 9-10 hold the lanthanides + actinides offset so
# they don't overlap the main table.  Columns 1-18 are groups.
PERIODIC_LAYOUT: dict[str, tuple[int, int]] = {
    "H":  (1,  1),  "He": (1, 18),
    "Li": (2,  1), "Be": (2,  2),
    "B":  (2, 13), "C":  (2, 14), "N":  (2, 15), "O":  (2, 16), "F":  (2, 17), "Ne": (2, 18),
    "Na": (3,  1), "Mg": (3,  2),
    "Al": (3, 13), "Si": (3, 14), "P":  (3, 15), "S":  (3, 16), "Cl": (3, 17), "Ar": (3, 18),
    "K":  (4,  1), "Ca": (4,  2),
    "Sc": (4,  3), "Ti": (4,  4), "V":  (4,  5), "Cr": (4,  6), "Mn": (4,  7), "Fe": (4,  8),
    "Co": (4,  9), "Ni": (4, 10), "Cu": (4, 11), "Zn": (4, 12),
    "Ga": (4, 13), "Ge": (4, 14), "As": (4, 15), "Se": (4, 16), "Br": (4, 17), "Kr": (4, 18),
    "Rb": (5,  1), "Sr": (5,  2),
    "Y":  (5,  3), "Zr": (5,  4), "Nb": (5,  5), "Mo": (5,  6), "Tc": (5,  7), "Ru": (5,  8),
    "Rh": (5,  9), "Pd": (5, 10), "Ag": (5, 11), "Cd": (5, 12),
    "In": (5, 13), "Sn": (5, 14), "Sb": (5, 15), "Te": (5, 16), "I":  (5, 17), "Xe": (5, 18),
    "Cs": (6,  1), "Ba": (6,  2),
    "Hf": (6,  4), "Ta": (6,  5), "W":  (6,  6), "Re": (6,  7), "Os": (6,  8),
    "Ir": (6,  9), "Pt": (6, 10), "Au": (6, 11), "Hg": (6, 12),
    "Tl": (6, 13), "Pb": (6, 14), "Bi": (6, 15), "Po": (6, 16), "At": (6, 17), "Rn": (6, 18),
    "Fr": (7,  1), "Ra": (7,  2),
    "Rf": (7,  4), "Db": (7,  5), "Sg": (7,  6), "Bh": (7,  7), "Hs": (7,  8),
    "Mt": (7,  9), "Ds": (7, 10), "Rg": (7, 11), "Cn": (7, 12),
    "Nh": (7, 13), "Fl": (7, 14), "Mc": (7, 15), "Lv": (7, 16), "Ts": (7, 17), "Og": (7, 18),
    # Lanthanides + actinides: offset two rows below the main grid for
    # visual separation, occupying columns 3..17 (15 elements each).
    "La": (9,  3), "Ce": (9,  4), "Pr": (9,  5), "Nd": (9,  6), "Pm": (9,  7),
    "Sm": (9,  8), "Eu": (9,  9), "Gd": (9, 10), "Tb": (9, 11), "Dy": (9, 12),
    "Ho": (9, 13), "Er": (9, 14), "Tm": (9, 15), "Yb": (9, 16), "Lu": (9, 17),
    "Ac": (10, 3), "Th": (10, 4), "Pa": (10, 5), "U":  (10, 6), "Np": (10, 7),
    "Pu": (10, 8), "Am": (10, 9), "Cm": (10,10), "Bk": (10,11), "Cf": (10,12),
    "Es": (10,13), "Fm": (10,14), "Md": (10,15), "No": (10,16), "Lr": (10,17),
}

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)\d*")


def count_elements(cif_dir: Path) -> tuple[Counter, int]:
    cifs = sorted(cif_dir.glob("*.cif"))
    counts: Counter[str] = Counter()
    for p in cifs:
        formula = p.stem.split("_", 1)[1] if "_" in p.stem else p.stem
        for el in set(_FORMULA_RE.findall(formula)):
            if el in atomic_numbers:
                counts[el] += 1
    return counts, len(cifs)


def main() -> None:
    cif_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else CIF_DIR
    out_path = cif_dir / "periodic_table_coverage.png"
    counts, n_total = count_elements(cif_dir)
    print(f"Scanned {n_total} CIFs in {cif_dir}; {len(counts)} unique elements present.")

    n_rows = max(r for r, _ in PERIODIC_LAYOUT.values())
    n_cols = 18

    fig, ax = plt.subplots(figsize=(18, 9))

    # Color scale: log-normalized because counts span 0..112 (~2 orders).
    vmax = max(counts.values()) if counts else 1
    norm = LogNorm(vmin=1, vmax=max(vmax, 2))
    cmap = plt.get_cmap("viridis")

    for sym, (row, col) in PERIODIC_LAYOUT.items():
        n = counts.get(sym, 0)
        # Bottom-up rendering: invert row so period 1 is at top.
        y = n_rows - row + 1
        x = col

        if n == 0:
            facecolor = "0.92"
            edgecolor = "0.7"
            text_color = "0.5"
        else:
            facecolor = cmap(norm(n))
            edgecolor = "black"
            # White text on dark cells, dark text on bright cells.
            text_color = "white" if norm(n) > 0.55 else "black"

        ax.add_patch(plt.Rectangle(
            (x - 0.45, y - 0.45), 0.9, 0.9,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5,
        ))
        # Atomic number (top-left, small).
        z = atomic_numbers[sym]
        ax.text(x - 0.38, y + 0.30, str(z), fontsize=6, color=text_color,
                ha="left", va="center")
        # Element symbol (centre, big).
        ax.text(x, y + 0.05, sym, fontsize=12, color=text_color,
                ha="center", va="center", weight="bold")
        # Count (bottom, medium).
        label = str(n) if n > 0 else "·"
        ax.text(x, y - 0.28, label, fontsize=9, color=text_color,
                ha="center", va="center")

    # Annotate the lanthanide / actinide rows.
    ax.text(2, n_rows - 9 + 1, "Lanthanides", fontsize=9, color="0.4",
            ha="center", va="center", style="italic")
    ax.text(2, n_rows - 10 + 1, "Actinides", fontsize=9, color="0.4",
            ha="center", va="center", style="italic")

    ax.set_xlim(0.5, n_cols + 0.5)
    ax.set_ylim(0.4, n_rows + 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    n_present = len(counts)
    n_absent = sum(1 for s in PERIODIC_LAYOUT if counts.get(s, 0) == 0)
    title = (
        f"Periodic-table coverage of selected CIFs\n"
        f"{cif_dir}  ·  {n_total} CIFs  ·  {n_present} elements present, "
        f"{n_absent} absent"
    )
    ax.set_title(title, fontsize=12)

    # Colorbar.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=20)
    cbar.set_label("# CIFs containing element  (log scale)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
