"""Analyze pilot output from compare_mace_teachers.py.

Reads summary.csv + the per-trajectory NPZ files and produces:

  - loss_curves.png  — overlay MPA vs OMAT loss curves, one subplot per system.
                       Healthy run: monotone descent; pathological run: plateau
                       early or NaN spike.
  - MO_distribution.png — histogram of metal–oxygen bond lengths at the final
                          frame, per (system, teacher).  Hubbard-U pathology
                          signature: MPA's distribution is shifted toward
                          longer bonds vs OMAT.
  - fmax_curves.png — fmax over FIRE steps, overlay MPA vs OMAT per system.
                      Lets you see if one teacher has trouble converging.
  - report.txt       — text summary of the comparison: per-system mean metrics
                       + a verdict ("keep with current teacher", "switch to
                       OMAT", "exclude").

Run with:
    /home/ehrdt/miniforge3/envs/mace/bin/python \\
        scripts/macerelax/pilot/analyze_teachers.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

OUTPUT_ROOT = Path("/home/ehrdt/tricor/mace/data/pilot_teacher_comparison")

# Heuristics for the verdict (adjust after seeing the data).
# "Tolerable" thresholds: how far apart MPA and OMAT can be on the same metric
# before we declare a meaningful difference.
TOLERANCE_M_O_BOND = 0.05    # Å — anything ≥ this is meaningful for an oxide
TOLERANCE_FINAL_LOSS_REL = 0.10   # 10% — final loss differs by more

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import csv
import json
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_summary(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def load_trajectory_npz(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        return {
            "positions": z["positions"],
            "loss":      z["loss"],
            "fmax":      z["fmax"],
            "species":   z["species"],
            "cell":      z["cell"],
            "meta":      json.loads(str(z["meta"])),
        }


def find_trajectory_files(root: Path, teacher: str, system: str) -> list[Path]:
    return sorted((root / teacher).glob(f"{system}_seed*.npz"))


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(systems: list[str], teachers: list[str],
                      out_path: Path) -> None:
    n = len(systems)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)
    colors = {"mpa": "#1f77b4", "omat": "#d62728"}
    for i, sys in enumerate(systems):
        ax = axes[i // ncols, i % ncols]
        for teacher in teachers:
            paths = find_trajectory_files(OUTPUT_ROOT, teacher, sys)
            for j, p in enumerate(paths):
                data = load_trajectory_npz(p)
                loss = data["loss"]
                # Shift to zero at start so we can compare *change* in loss.
                loss_shift = loss - loss[0]
                ax.plot(loss_shift,
                        color=colors.get(teacher, "gray"),
                        alpha=0.7,
                        label=teacher if j == 0 else None)
        ax.set_title(sys)
        ax.set_xlabel("FIRE step")
        ax.set_ylabel("ΔE (eV)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle("Loss curves — MPA vs OMAT per system "
                 "(ΔE = E(step) − E(step=0))")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_fmax_curves(systems: list[str], teachers: list[str],
                      out_path: Path) -> None:
    n = len(systems)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)
    colors = {"mpa": "#1f77b4", "omat": "#d62728"}
    for i, sys in enumerate(systems):
        ax = axes[i // ncols, i % ncols]
        for teacher in teachers:
            paths = find_trajectory_files(OUTPUT_ROOT, teacher, sys)
            for j, p in enumerate(paths):
                data = load_trajectory_npz(p)
                ax.semilogy(data["fmax"],
                            color=colors.get(teacher, "gray"),
                            alpha=0.7,
                            label=teacher if j == 0 else None)
        ax.set_title(sys)
        ax.set_xlabel("FIRE step")
        ax.set_ylabel("fmax (eV/Å, log)")
        ax.grid(True, alpha=0.3, which="both")
        if i == 0:
            ax.legend(loc="best")
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle("|F|max trajectories — MPA vs OMAT per system")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_MO_distribution(systems: list[str], teachers: list[str],
                          out_path: Path) -> None:
    """Histogram of all metal–oxygen distances at final frame, per teacher.

    Hubbard-U pathology signature: MPA's distribution shifted toward longer
    M–O bonds vs OMAT.  Plot only U-affected oxide systems.
    """
    U_METALS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Mo", "W"}

    # Determine which systems have a U-affected metal + O.
    relevant = []
    for sys in systems:
        for teacher in teachers:
            paths = find_trajectory_files(OUTPUT_ROOT, teacher, sys)
            if not paths:
                continue
            data = load_trajectory_npz(paths[0])
            species = set(data["species"].tolist())
            if (species & U_METALS) and "O" in species:
                relevant.append(sys)
                break
    if not relevant:
        print("  [skip] no U-affected M–O systems to plot")
        return

    n = len(relevant)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              squeeze=False)
    colors = {"mpa": "#1f77b4", "omat": "#d62728"}
    for i, sys in enumerate(relevant):
        ax = axes[i // ncols, i % ncols]
        for teacher in teachers:
            paths = find_trajectory_files(OUTPUT_ROOT, teacher, sys)
            all_bonds = []
            for p in paths:
                data = load_trajectory_npz(p)
                pos = data["positions"][-1]   # final frame
                species = data["species"]
                cell = data["cell"]
                bonds = _all_MO_bonds(pos, species, cell, U_METALS)
                all_bonds.extend(bonds)
            if all_bonds:
                ax.hist(all_bonds, bins=30, range=(1.5, 3.0),
                        color=colors.get(teacher, "gray"),
                        alpha=0.5, label=teacher,
                        density=True)
        ax.set_title(sys)
        ax.set_xlabel("M–O distance (Å)")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle("Final-frame M–O bond distribution "
                 "(longer = U pathology signature)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _all_MO_bonds(pos: np.ndarray, species: np.ndarray, cell: np.ndarray,
                   metals: set, max_r: float = 3.0) -> list[float]:
    """All metal–O distances ≤ max_r under PBC."""
    metal_idx  = np.where(np.isin(species, list(metals)))[0]
    oxygen_idx = np.where(species == "O")[0]
    if len(metal_idx) == 0 or len(oxygen_idx) == 0:
        return []
    inv_cell = np.linalg.inv(cell)
    bonds = []
    for mi in metal_idx:
        deltas = pos[oxygen_idx] - pos[mi]
        frac = deltas @ inv_cell
        frac = frac - np.round(frac)
        deltas = frac @ cell
        dists = np.linalg.norm(deltas, axis=1)
        bonds.extend(dists[dists <= max_r].tolist())
    return bonds


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(summary_rows: list[dict], out_path: Path) -> None:
    """Per-system MPA-vs-OMAT verdict."""
    by_sys_teacher = defaultdict(list)
    for r in summary_rows:
        by_sys_teacher[(r["system"], r["teacher"])].append(r)

    systems = sorted({r["system"] for r in summary_rows})

    with open(out_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MACE-MPA vs MACE-OMAT teacher comparison report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Tolerances:\n")
        f.write(f"  M-O bond difference  : {TOLERANCE_M_O_BOND} Å\n")
        f.write(f"  final-loss difference: {TOLERANCE_FINAL_LOSS_REL * 100:.0f}%\n\n")

        for sys in systems:
            f.write(f"\n--- {sys} ---\n")
            metrics = {}
            for teacher in ("mpa", "omat"):
                rows = by_sys_teacher.get((sys, teacher), [])
                if not rows:
                    f.write(f"  {teacher}: no data\n")
                    metrics[teacher] = None
                    continue
                is_u = rows[0]["is_u_affected"] == "True"
                n_conv = sum(1 for r in rows if r["converged"] == "True")
                final_loss = np.nanmean([float(r["final_loss"]) for r in rows])
                fmax_final = np.nanmean([float(r["fmax_final"]) for r in rows])
                MO_vals = [float(r["mean_M_O_bond"]) for r in rows
                            if r["mean_M_O_bond"] not in ("", "None", None)]
                mean_MO = np.mean(MO_vals) if MO_vals else None
                f.write(f"  {teacher}: "
                        f"converged={n_conv}/{len(rows)} | "
                        f"final_loss={final_loss:.3e} | "
                        f"fmax_final={fmax_final:.2e} | "
                        f"mean_M-O={mean_MO}\n")
                metrics[teacher] = dict(
                    is_u=is_u, mean_MO=mean_MO, final_loss=final_loss,
                )

            # Verdict.
            mpa  = metrics.get("mpa")
            omat = metrics.get("omat")
            if not mpa or not omat:
                f.write("  → verdict: skipped (incomplete data)\n")
                continue
            if not mpa["is_u"]:
                f.write("  → verdict: not U-affected — either teacher OK\n")
                continue

            # The signature: OMAT's M-O bond should be shorter (closer to DFT)
            # if MACE-MPA has the pathology.
            verdicts = []
            if mpa["mean_MO"] is not None and omat["mean_MO"] is not None:
                d_MO = mpa["mean_MO"] - omat["mean_MO"]
                if d_MO >= TOLERANCE_M_O_BOND:
                    verdicts.append(
                        f"MPA's mean M-O is {d_MO:.3f} Å LONGER than OMAT's "
                        f"— Hubbard-U pathology signature confirmed")
                elif d_MO <= -TOLERANCE_M_O_BOND:
                    verdicts.append(
                        f"MPA's mean M-O is {abs(d_MO):.3f} Å SHORTER than "
                        f"OMAT's — unexpected, investigate")
                else:
                    verdicts.append(
                        f"MPA and OMAT M-O bond lengths agree to "
                        f"{abs(d_MO):.3f} Å — no detectable pathology")
            f_diff_rel = (omat["final_loss"] - mpa["final_loss"]) / abs(mpa["final_loss"])
            if abs(f_diff_rel) > TOLERANCE_FINAL_LOSS_REL:
                verdicts.append(
                    f"final-loss differs by {f_diff_rel * 100:.0f}% "
                    f"(OMAT vs MPA)")

            f.write(f"  → verdict: {' | '.join(verdicts)}\n")

        f.write("\n" + "=" * 70 + "\n")
    print(f"  wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    summary_path = OUTPUT_ROOT / "summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"[abort] {summary_path} not found — run "
                          f"compare_mace_teachers.py first")
    summary = load_summary(summary_path)
    systems  = sorted({r["system"] for r in summary})
    teachers = sorted({r["teacher"] for r in summary})
    print(f"[analyze] {len(summary)} trajectories across "
          f"{len(systems)} systems × {len(teachers)} teachers")

    plot_loss_curves(systems, teachers, OUTPUT_ROOT / "loss_curves.png")
    plot_fmax_curves(systems, teachers, OUTPUT_ROOT / "fmax_curves.png")
    plot_MO_distribution(systems, teachers, OUTPUT_ROOT / "MO_distribution.png")
    write_report(summary, OUTPUT_ROOT / "report.txt")
    print(f"[analyze] outputs at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
