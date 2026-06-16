"""Side-by-side comparison: fast vs slow ``_build_grain_atoms``.

Generates a supercell for each regime via BOTH paths (toggled via
``Supercell._USE_FAST_GRAIN_PATH``), writes both as XYZ files for
visual inspection, and prints quantitative diffs:

  * Total atom count
  * Per-species atom count
  * Radial distribution function g(r) on a coarse grid
  * Wall-clock per path

Same RNG seed for both paths.  The two structures will NOT be
bit-identical because the slow path consumes random numbers for
Voronoi-cell padding that the fast path skips — but they should be
*statistically* equivalent: same atom counts, same per-species
counts, same first-shell g(r) peaks.

Outputs land in ``scratch/validate_xyz/<cif>/<regime>_{fast,slow}.xyz``.

CPU-only, no model.  Production cell size (100×100×400) is too slow
for the slow path (~3 min/regime), so the default is 50³ — bump
``CELL_DIMS`` if you want a closer-to-production comparison.

Run:
    python scratch/validate_fast_path.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read as ase_read, write as ase_write

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import tricor as tc                                             # noqa: E402
from tricor import CoordinationShellTarget, G3Distribution      # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CIF_DIR = Path("/home/ehrdt/cifs_mp_exp_le100meV")

# Mostly Si (fastest reference) — add the others if you want denser chem
# checks, but keep in mind the slow path scales as ~(box volume / grain
# volume) Python iterations.
CIF_FILES = [
    "mp-149_Si.cif",        # single-species sp3 (reference)
    "mp-1439_TiO2.cif",     # two-species oxide
    "mp-21476_Fe2N.cif",    # dense metal nitride
]

# All 6 regimes; comment out individual entries to limit runtime.
REGIMES = ["amorphous", "SRO", "MRO", "LRO", "nanocrystalline", "crystalline_30"]

DENSITY_BY_REGIME = {
    "amorphous":       0.92,
    "SRO":             0.92,
    "MRO":             0.88,
    "LRO":             0.92,
    "nanocrystalline": 0.96,
    "crystalline_30":  0.98,
}

# 50³ keeps the slow path tractable on amorphous (~30 s vs ~3 min at
# 100×100×400).  Bump if you want a stress test closer to production.
CELL_DIMS = (50.0, 50.0, 50.0)
RNG_SEED  = 2_000_000

BOND_RELAX_N_ITER   = 20
BOND_RELAX_MAX_STEP = 0.1
PHI_NUM_BINS        = 90

# g(r) histogram parameters for the structural comparison.
RDF_R_MAX  = 8.0
RDF_NBINS  = 80

OUT_DIR = _REPO / "scratch" / "validate_xyz"


# ─────────────────────────────────────────────────────────────────────────────
# Preset assembly (mirrors generate_with_student.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_local_presets() -> dict:
    base: dict = {}
    for name in REGIMES:
        if name == "crystalline_30":
            base[name] = dict(
                num_steps=0, grain_size=30.0, displacement_sigma=0.0,
                bond_weight=3.0, angle_weight=1.5,
            )
        else:
            d = dict(tc.Supercell.PRESETS[name])
            d["displacement_sigma"] = 0.0
            base[name] = d
    return base


LOCAL_PRESETS = _build_local_presets()


# ─────────────────────────────────────────────────────────────────────────────
# Single-regime build (fast or slow)
# ─────────────────────────────────────────────────────────────────────────────

def build(cif_path: Path, regime: str, use_fast: bool):
    """Build a supercell for one regime under the chosen code path."""
    # The escape-hatch flag is read by _build_grain_atoms at dispatch
    # time; set it BEFORE generate() runs.
    tc.Supercell._USE_FAST_GRAIN_PATH = bool(use_fast)

    preset = dict(LOCAL_PRESETS[regime])
    preset["num_steps"] = 0
    rho = DENSITY_BY_REGIME[regime]

    ref = ase_read(str(cif_path), format="cif")
    shell = CoordinationShellTarget.from_atoms(ref, phi_num_bins=PHI_NUM_BINS)
    dist = G3Distribution(ref, label=cif_path.stem)
    dist.measure_g3(r_max=10.0, r_step=0.1, phi_num_bins=PHI_NUM_BINS,
                    show_progress=False)

    cell = tc.Supercell(
        dist,
        cell_dim_angstroms=tuple(CELL_DIMS),
        relative_density=rho,
        rng_seed=RNG_SEED,
        label=f"{cif_path.stem}_{regime}_{RNG_SEED}",
    )
    t0 = time.perf_counter()
    summary = cell.generate(
        shell, **preset, refine_orientations=False, show_progress=False,
    )
    t_pack = time.perf_counter() - t0

    t0 = time.perf_counter()
    cell.bond_relax(shell, n_iter=BOND_RELAX_N_ITER,
                    max_step=BOND_RELAX_MAX_STEP)
    t_cleanup = time.perf_counter() - t0

    return cell, summary, t_pack, t_cleanup


# ─────────────────────────────────────────────────────────────────────────────
# Diff helpers
# ─────────────────────────────────────────────────────────────────────────────

def per_species_counts(atoms) -> dict[int, int]:
    nums = atoms.numbers
    return {int(z): int((nums == z).sum()) for z in np.unique(nums)}


def rdf(atoms, r_max: float = RDF_R_MAX, nbins: int = RDF_NBINS) -> np.ndarray:
    """Return a normalized g(r) over ``nbins`` linear bins in [0, r_max].

    Sum-of-pairs only — no species splitting.  Enough to compare bulk
    structure between the two paths.
    """
    from ase.neighborlist import neighbor_list
    if len(atoms) < 2:
        return np.zeros(nbins)
    # ase's neighbor_list returns all pairs within r_max with both orderings.
    _, _, d = neighbor_list("ijd", atoms, r_max)
    hist, edges = np.histogram(d, bins=nbins, range=(0.0, r_max))
    r = 0.5 * (edges[:-1] + edges[1:])
    # Normalize to g(r): hist / (4 pi r^2 dr * N * rho)
    N = len(atoms)
    V = float(atoms.cell.volume)
    rho = N / max(V, 1e-12)
    dr = edges[1] - edges[0]
    norm = 4.0 * np.pi * r ** 2 * dr * N * rho
    g = np.where(norm > 0, hist.astype(float) / norm, 0.0)
    return g, r


def rdf_l2(g1: np.ndarray, g2: np.ndarray) -> float:
    """Relative L2 difference between two g(r) curves."""
    num = float(np.sqrt(np.sum((g1 - g2) ** 2)))
    den = float(np.sqrt(np.sum((g1 + g2) ** 2)) + 1e-12)
    return num / den


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[validate] cell={CELL_DIMS}  regimes={REGIMES}")
    print(f"[validate] XYZ output → {OUT_DIR}")
    for cif_name in CIF_FILES:
        cif_path = CIF_DIR / cif_name
        if not cif_path.is_file():
            print(f"  ✗ missing: {cif_path}")
            continue
        sub = OUT_DIR / cif_path.stem
        sub.mkdir(exist_ok=True)
        print(f"\n━━━ {cif_name} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        for regime in REGIMES:
            print(f"\n  [{regime}]")
            try:
                cell_f, sum_f, tp_f, tc_f = build(cif_path, regime,
                                                 use_fast=True)
            except Exception as exc:
                print(f"    fast FAILED — {type(exc).__name__}: {exc}")
                continue
            try:
                cell_s, sum_s, tp_s, tc_s = build(cif_path, regime,
                                                 use_fast=False)
            except Exception as exc:
                print(f"    slow FAILED — {type(exc).__name__}: {exc}")
                continue

            # Write XYZ files.
            f_xyz = sub / f"{regime}_fast.xyz"
            s_xyz = sub / f"{regime}_slow.xyz"
            ase_write(str(f_xyz), cell_f.atoms, format="extxyz")
            ase_write(str(s_xyz), cell_s.atoms, format="extxyz")

            # Quantitative diff.
            nf, ns = len(cell_f.atoms), len(cell_s.atoms)
            zsf = per_species_counts(cell_f.atoms)
            zss = per_species_counts(cell_s.atoms)
            gf, r = rdf(cell_f.atoms)
            gs, _ = rdf(cell_s.atoms)
            l2 = rdf_l2(gf, gs)
            peak_f = float(r[int(np.argmax(gf))])
            peak_s = float(r[int(np.argmax(gs))])

            print(f"    fast: pack={tp_f:5.1f}s cleanup={tc_f:4.1f}s "
                  f"n={nf:>6}  per-Z={zsf}")
            print(f"    slow: pack={tp_s:5.1f}s cleanup={tc_s:4.1f}s "
                  f"n={ns:>6}  per-Z={zss}")
            print(f"    Δn = {nf-ns:+d}  ({100*(nf-ns)/max(ns,1):+.2f}%)  "
                  f"speedup = {tp_s/max(tp_f,1e-3):.1f}×")
            print(f"    g(r) L2 = {l2:.4f}   "
                  f"first peak: fast={peak_f:.2f}Å  slow={peak_s:.2f}Å")
            print(f"    XYZ: {f_xyz.relative_to(_REPO)}")
            print(f"         {s_xyz.relative_to(_REPO)}")

    print("\n[validate] done.")
    print("[validate] Open the XYZ pairs in OVITO (or your viewer of "
          "choice) and check that fast/slow show the same packing "
          "character per regime.")


if __name__ == "__main__":
    main()
