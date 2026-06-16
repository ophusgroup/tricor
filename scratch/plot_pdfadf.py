"""Throwaway: PDF/ADF comparison across three LAMMPS MD snapshots.

Mirrors the 2-panel g(r) + ADF style of
~/packages/tricor/scripts/relaxml/evaluate.py, but extended to SiO2's
multiple pair/triplet types and to three structures overlaid instead of
target-vs-predicted.

Run from anywhere — paths are absolute. Output saved next to the data.
"""
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / "packages/tricor/src"))
from tricor.differentiable_pdf_fast import DifferentiablePDFADF_Fast

DATA_DIR = Path.home() / "data/tricor_lammps"
# Per-regime snapshot files (file naming was inconsistent across regimes,
# hence explicit triples rather than a single suffix pattern).
REGIMES = {
    "nanocrystalline": ["sio2_before_lmp", "sio2_1step", "sio2_final_lmp"],
    "amorphous": ["amorphous_before_lammps", "amorphous_1step", "amorphous_lmp_final"],
    "MRO":       ["MRO_before_lammps",       "MRO_1step",       "MRO_lmp_final"],
}
LABELS = ["before LAMMPS", "1 step", "final"]
COLORS = ["0.55", "C0", "C3"]

# Same PDF/ADF resolution as evaluate.py, but r_max bumped up so the
# regime-comparison plot can show medium-range order out past the 8 Å
# first-few-shells region. (Crystalline grains, if present, give sharper
# peaks at larger r than a fully amorphous structure does.)
PDF_R_MAX  = 18.0
PDF_R_STEP = 0.05
PDF_PHI_BINS = 90
PLOT_R_MAX = 8.0
COMPARE_R_MAX = 16.0  # x-axis limit on the regime-comparison panel
# Tight ADF cutoff so only first-shell (bonded) triplets contribute.
# Si–O peak is ~1.6 Å; O–O and Si–Si start near 2.5–3.1 Å. 2.2 Å keeps
# just the Si–O bond, giving clean O–Si–O (tetrahedral, ~109°) and
# Si–O–Si (bridging, ~140°) ADFs; broader cutoffs wash these into sin(φ).
ADF_R_MAX = 2.2

# File species column: "1" → Si (Z=14), "2" → O (Z=8). Ratio in
# sio2_before_lmp is 1619:3268 ≈ 1:2, matching SiO2 stoichiometry.
FILE_SPECIES_TO_Z = {"1": 14, "2": 8}
SPECIES = sorted(set(FILE_SPECIES_TO_Z.values()))   # [8, 14] → idx 0=O, 1=Si


def parse_extxyz(path: Path):
    with path.open() as fh:
        n = int(fh.readline().strip())
        header = fh.readline()
        lat_start = header.index('Lattice="') + len('Lattice="')
        lat_end = header.index('"', lat_start)
        cell = np.array(
            [float(x) for x in header[lat_start:lat_end].split()],
            dtype=np.float64,
        ).reshape(3, 3)
        positions = np.empty((n, 3), dtype=np.float64)
        z = np.empty(n, dtype=np.int64)
        for i in range(n):
            parts = fh.readline().split()
            positions[i] = [float(x) for x in parts[:3]]
            z[i] = FILE_SPECIES_TO_Z[parts[3]]
    return positions, z, cell


def compute_pdf_adf(positions, species, cell):
    mod = DifferentiablePDFADF_Fast(
        r_max=PDF_R_MAX, r_step=PDF_R_STEP,
        phi_num_bins=PDF_PHI_BINS, species=SPECIES,
        adf_r_max=ADF_R_MAX,
    ).to(torch.float64)
    with torch.no_grad():
        g2, adf = mod.compute(
            torch.tensor(positions, dtype=torch.float64),
            torch.tensor(species, dtype=torch.int64),
            torch.tensor(cell, dtype=torch.float64),
        )
    return g2.numpy(), adf.numpy()


def grids():
    num_r = int(round(PDF_R_MAX / PDF_R_STEP))
    r = np.arange(num_r) * PDF_R_STEP + 0.5 * PDF_R_STEP
    phi_edges = np.linspace(0.0, np.pi, PDF_PHI_BINS + 1)
    phi = phi_edges[:-1] + 0.5 * (phi_edges[1] - phi_edges[0])
    return r, np.rad2deg(phi)


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def gofr(g_pair, ni, nj, V, r):
    denom = ni * (ni - 1) if ni == nj else ni * nj
    if denom <= 0:
        return g_pair * 0.0
    shell = 4.0 * np.pi * np.maximum(r, 1e-6) ** 2 * PDF_R_STEP
    return g_pair * V / (denom * shell)


def area_norm(y, x):
    area = float(_trapz(y, x))
    return y / area if area > 0 else y


def plot_regime(regime: str, files: list[str], r_grid, phi_grid) -> Path:
    print(f"[{regime}]")
    snaps = []
    for name in files:
        pos, z, cell = parse_extxyz(DATA_DIR / name)
        g2, adf = compute_pdf_adf(pos, z, cell)
        snaps.append({
            "name": name, "z": z, "cell": cell, "g2": g2, "adf": adf,
            "V": float(abs(np.linalg.det(cell))),
        })
        print(f"  {name}: N={len(pos)}  Si={(z==14).sum()}  O={(z==8).sum()}  "
              f"cell_diag={tuple(np.round(np.diag(cell),3))}")

    O_i, Si_i = 0, 1   # sorted([8,14]) → O first
    pair_panels = [
        ("Si–Si", Si_i, Si_i),
        ("Si–O",  Si_i, O_i),
        ("O–O",   O_i,  O_i),
    ]
    # g3_index order for sorted species [O, Si]:
    #   0: O–O–O   1: O–O–Si  2: O–Si–Si
    #   3: Si–O–O  4: Si–O–Si 5: Si–Si–Si
    adf_panels = [
        ("ADF — Si-centered", [3, 4, 5]),
        ("ADF — O-centered",  [0, 1, 2]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for col, (label, i, j) in enumerate(pair_panels):
        ax = axes[0, col]
        for snap, slabel, color in zip(snaps, LABELS, COLORS):
            ni = int((snap["z"] == SPECIES[i]).sum())
            nj = int((snap["z"] == SPECIES[j]).sum())
            y = gofr(snap["g2"][i, j], ni, nj, snap["V"], r_grid)
            ax.plot(r_grid, y, color=color, lw=1.6, label=slabel)
        ax.set_xlim(0.0, PLOT_R_MAX)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"PDF — {label}")
        ax.axhline(1.0, color="0.7", lw=0.7, ls=":")
        if col == 0:
            ax.legend(framealpha=0.9, fontsize=9)

    for col, (title, idxs) in enumerate(adf_panels):
        ax = axes[1, col]
        for snap, slabel, color in zip(snaps, LABELS, COLORS):
            adf_sum = snap["adf"][idxs].sum(axis=0)
            ax.plot(phi_grid, area_norm(adf_sum, phi_grid),
                    color=color, lw=1.6, label=slabel)
        ax.set_xlabel("bond angle φ (deg)")
        ax.set_ylabel("ADF(φ)  [normalized]")
        ax.set_title(title)
        if col == 0:
            ax.legend(framealpha=0.9, fontsize=9)
    axes[1, 2].axis("off")

    fig.suptitle(f"SiO2 [{regime}] — PDF & ADF across MD snapshots", fontsize=12)
    fig.tight_layout()
    out = DATA_DIR / f"snapshots_pdf_adf_{regime}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved → {out}")
    return out


def compare_final_states(r_grid, phi_grid) -> Path:
    """Overlay the final (LAMMPS-relaxed) structure from each regime.

    Specifically aimed at answering: does the nanocrystalline regime
    retain crystalline order after MD, or does it collapse to amorphous?
    Signature of retained crystallinity is sharper PDF peaks at larger
    r (medium-range order) than the amorphous final state.
    """
    regime_colors = {
        "nanocrystalline": "C2",
        "MRO":             "C1",
        "amorphous":       "C0",
    }
    print("[compare]")
    finals = {}
    for regime, files in REGIMES.items():
        final_file = files[-1]
        pos, z, cell = parse_extxyz(DATA_DIR / final_file)
        g2, adf = compute_pdf_adf(pos, z, cell)
        finals[regime] = {
            "z": z, "cell": cell, "g2": g2, "adf": adf,
            "V": float(abs(np.linalg.det(cell))),
        }
        print(f"  {regime}: {final_file}  N={len(pos)}")

    O_i, Si_i = 0, 1
    pair_panels = [("Si–Si", Si_i, Si_i), ("Si–O", Si_i, O_i), ("O–O", O_i, O_i)]
    adf_panels = [
        ("ADF — Si-centered", [3, 4, 5]),
        ("ADF — O-centered",  [0, 1, 2]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for col, (label, i, j) in enumerate(pair_panels):
        ax = axes[0, col]
        for regime, snap in finals.items():
            ni = int((snap["z"] == SPECIES[i]).sum())
            nj = int((snap["z"] == SPECIES[j]).sum())
            y = gofr(snap["g2"][i, j], ni, nj, snap["V"], r_grid)
            ax.plot(r_grid, y, color=regime_colors[regime], lw=1.4, label=regime)
        ax.set_xlim(0.0, COMPARE_R_MAX)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"PDF — {label}")
        ax.axhline(1.0, color="0.7", lw=0.7, ls=":")
        if col == 0:
            ax.legend(framealpha=0.9, fontsize=9)

    for col, (title, idxs) in enumerate(adf_panels):
        ax = axes[1, col]
        for regime, snap in finals.items():
            adf_sum = snap["adf"][idxs].sum(axis=0)
            ax.plot(phi_grid, area_norm(adf_sum, phi_grid),
                    color=regime_colors[regime], lw=1.4, label=regime)
        ax.set_xlabel("bond angle φ (deg)")
        ax.set_ylabel("ADF(φ)  [normalized]")
        ax.set_title(title)
        if col == 0:
            ax.legend(framealpha=0.9, fontsize=9)
    axes[1, 2].axis("off")

    fig.suptitle("LAMMPS-final structure compared across regimes "
                 "(crystallinity test: look for sharp medium-range PDF peaks)",
                 fontsize=11)
    fig.tight_layout()
    out = DATA_DIR / "snapshots_pdf_adf_compare_final.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved → {out}")
    return out


def main():
    r_grid, phi_grid = grids()
    for regime, files in REGIMES.items():
        plot_regime(regime, files, r_grid, phi_grid)
    compare_final_states(r_grid, phi_grid)


if __name__ == "__main__":
    main()
