"""Re-plot scale_invariance_data.npz at a readable size.

Produces THREE PNGs:

  1. scale_invariance_by_system_<label>.png  (one per system)
     Per-system grid: 6 rows (regimes) × 2 cols (g(r) | ADF), all 5
     curves overlaid.  Panels are ~5 × 3.5 inches each — large enough
     to read peak positions.

  2. scale_invariance_init_overlap.png
     Focused view: ONLY the small-init vs large-init curves, for all
     3 systems in one figure.  6 rows × 6 cols (3 systems × g(r)|ADF),
     larger panels.  Tells you at a glance whether your bond_relax
     pipeline scales correctly.

  3. scale_invariance_all_overlay.png  (the original 6×6 grid, bigger
     panels — kept around for the full picture).

Run from anywhere:
    /home/ehrdt/miniforge3/envs/mace/bin/python scratch/replot_scale_invariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]

# Cache labels are the short names from CIFS in scale_invariance_check.py
SYSTEMS = ["Ho2C", "Tm2S3", "Pa"]
SYSTEM_LABELS = {"Ho2C": "Ho2C", "Tm2S3": "Tm2S3", "Pa": "Pa"}
REGIMES = ["amorphous", "SRO", "MRO", "LRO", "nanocrystalline", "crystalline_30"]

OUT_DIR   = _REPO / "scratch"
# scale_invariance_check.py auto-suffixes its cache by dtype:
#   "_fp32" if USE_BF16_INFERENCE = False
#   "_bf16" if USE_BF16_INFERENCE = True
#   ""     for legacy runs before the suffix was added
# Set DTYPE_SUFFIX to whichever run you want to plot.  All output PNGs
# get the same suffix so multiple runs never clobber each other.
DTYPE_SUFFIX = "_bf16"
CACHE_NPZ = OUT_DIR / f"scale_invariance_data{DTYPE_SUFFIX}.npz"

# alpha=0.7 lets overlaid curves all show through.  Pick distinct lw
# + linestyle so curves remain individually distinguishable even when
# perfectly stacked.
CURVE_STYLE = {
    "small_init":    dict(color="#888888", lw=1.8, ls="--", alpha=0.7,
                          label="small init"),
    "small_mace":    dict(color="#000000", lw=2.0, ls="-",  alpha=0.7,
                          label="small MACE"),
    "small_student": dict(color="#1f77b4", lw=1.8, ls="-",  alpha=0.7,
                          label="small student"),
    "large_init":    dict(color="#aaaaaa", lw=1.8, ls=":",  alpha=0.7,
                          label="large init"),
    "large_student": dict(color="#d62728", lw=1.8, ls="-",  alpha=0.7,
                          label="large student"),
}

INIT_ONLY_STYLE = {
    "small_init": dict(color="#1f77b4", lw=2.2, ls="-",  alpha=0.75,
                       label="small init"),
    "large_init": dict(color="#d62728", lw=2.2, ls="--", alpha=0.75,
                       label="large init"),
}

PDF_R_MAX = 8.0


# ─────────────────────────────────────────────────────────────────────────────
# Load cache
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict:
    blob = np.load(str(path), allow_pickle=True)
    out: dict = {}
    for key in blob.files:
        system_label, regime, name, arr = key.split("|")
        out.setdefault(system_label, {}).setdefault(regime, {}).setdefault(
            name, {})[arr] = blob[key]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — per-system full grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_system(all_curves: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for system_key in SYSTEMS:
        label = SYSTEM_LABELS[system_key]
        n_rows = len(REGIMES)
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(11, 2.5 * n_rows),
            squeeze=False, sharex="col",
        )
        for r_idx, regime in enumerate(REGIMES):
            ax_g = axes[r_idx, 0]
            ax_a = axes[r_idx, 1]
            curves = all_curves.get(system_key, {}).get(regime, {})
            for name, style in CURVE_STYLE.items():
                d = curves.get(name)
                if d is None:
                    continue
                ax_g.plot(d["r"], d["g"], **style)
                ax_a.plot(d["th"], d["ad"], **style)
            ax_g.set_xlim(0, PDF_R_MAX)
            ax_a.set_xlim(0, 180)
            ax_g.set_ylabel(f"{regime}\n\ng(r)", fontsize=11)
            ax_a.set_ylabel("P(θ)", fontsize=11)
            if r_idx == 0:
                ax_g.set_title(f"{label} — g(r)", fontsize=13)
                ax_a.set_title(f"{label} — ADF", fontsize=13)
            if r_idx == n_rows - 1:
                ax_g.set_xlabel("r (Å)", fontsize=11)
                ax_a.set_xlabel("θ (deg)", fontsize=11)
            ax_g.grid(alpha=0.3, lw=0.4)
            ax_a.grid(alpha=0.3, lw=0.4)
        # Legend at bottom
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=s["color"], lw=s["lw"], ls=s["ls"])
                   for s in CURVE_STYLE.values()]
        labels = [s["label"] for s in CURVE_STYLE.values()]
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(CURVE_STYLE), bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"{label} — scale invariance: small (50³) vs large "
                     f"(100×100×400)", fontsize=14, y=1.00)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.99))
        out_path = out_dir / f"scale_invariance_by_system_{label}{DTYPE_SUFFIX}.png"
        fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — init-only overlap, all systems
# ─────────────────────────────────────────────────────────────────────────────

def plot_init_overlap(all_curves: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(REGIMES)
    n_systems = len(SYSTEMS)
    fig, axes = plt.subplots(
        n_rows, 2 * n_systems,
        figsize=(4.0 * 2 * n_systems, 2.8 * n_rows),
        squeeze=False, sharex="col",
    )
    for r_idx, regime in enumerate(REGIMES):
        for s_idx, system_key in enumerate(SYSTEMS):
            label = SYSTEM_LABELS[system_key]
            ax_g = axes[r_idx, 2 * s_idx]
            ax_a = axes[r_idx, 2 * s_idx + 1]
            curves = all_curves.get(system_key, {}).get(regime, {})
            for name, style in INIT_ONLY_STYLE.items():
                d = curves.get(name)
                if d is None:
                    continue
                ax_g.plot(d["r"], d["g"], **style)
                ax_a.plot(d["th"], d["ad"], **style)
            ax_g.set_xlim(0, PDF_R_MAX)
            ax_a.set_xlim(0, 180)
            if r_idx == 0:
                ax_g.set_title(f"{label} — g(r)", fontsize=12)
                ax_a.set_title(f"{label} — ADF", fontsize=12)
            if s_idx == 0:
                ax_g.set_ylabel(regime, fontsize=11, rotation=0,
                                ha="right", va="center")
            if r_idx == n_rows - 1:
                ax_g.set_xlabel("r (Å)", fontsize=10)
                ax_a.set_xlabel("θ (deg)", fontsize=10)
            ax_g.grid(alpha=0.3, lw=0.4)
            ax_a.grid(alpha=0.3, lw=0.4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=s["color"], lw=s["lw"], ls=s["ls"])
               for s in INIT_ONLY_STYLE.values()]
    labels = [s["label"] for s in INIT_ONLY_STYLE.values()]
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), fontsize=12)
    fig.suptitle("Initial-structure scale invariance "
                 "(bond_relax pipeline only, no MACE / no student)",
                 fontsize=14, y=1.00)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.99))
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — full 6×6 overlay (larger than the run-script default)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_regime(all_curves: dict, out_dir: Path) -> None:
    """One figure per regime, showing all 3 systems side-by-side.

    Layout: 1 row × 6 cols = (Si g(r), Si ADF, TiO2 g(r), TiO2 ADF,
    Fe2N g(r), Fe2N ADF).  All 5 curves overlaid with alpha=0.7 so
    near-identical traces all stay visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_systems = len(SYSTEMS)
    for regime in REGIMES:
        fig, axes = plt.subplots(
            1, 2 * n_systems,
            figsize=(4.5 * 2 * n_systems, 4.5),
            squeeze=False, sharex="col",
        )
        for s_idx, system in enumerate(SYSTEMS):
            label = SYSTEM_LABELS[system]
            ax_g = axes[0, 2 * s_idx]
            ax_a = axes[0, 2 * s_idx + 1]
            curves = all_curves.get(system, {}).get(regime, {})
            for name, style in CURVE_STYLE.items():
                d = curves.get(name)
                if d is None:
                    continue
                ax_g.plot(d["r"], d["g"], **style)
                ax_a.plot(d["th"], d["ad"], **style)
            ax_g.set_xlim(0, PDF_R_MAX)
            ax_a.set_xlim(0, 180)
            ax_g.set_title(f"{label} — g(r)", fontsize=13)
            ax_a.set_title(f"{label} — ADF", fontsize=13)
            ax_g.set_xlabel("r (Å)", fontsize=11)
            ax_a.set_xlabel("θ (deg)", fontsize=11)
            ax_g.grid(alpha=0.3, lw=0.4)
            ax_a.grid(alpha=0.3, lw=0.4)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=s["color"], lw=s["lw"], ls=s["ls"],
                          alpha=s.get("alpha", 1.0))
                   for s in CURVE_STYLE.values()]
        labels = [s["label"] for s in CURVE_STYLE.values()]
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(CURVE_STYLE), bbox_to_anchor=(0.5, -0.02),
                   fontsize=11)
        fig.suptitle(f"Regime: {regime}  —  scale invariance (small vs large)",
                     fontsize=14, y=1.02)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.98))
        out_path = out_dir / f"scale_invariance_by_regime_{regime}{DTYPE_SUFFIX}.png"
        fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}", flush=True)


def plot_full_overlay(all_curves: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(REGIMES)
    n_systems = len(SYSTEMS)
    n_cols = n_systems * 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.0 * n_rows),
        squeeze=False, sharex="col",
    )
    for r_idx, regime in enumerate(REGIMES):
        for s_idx, system_key in enumerate(SYSTEMS):
            label = SYSTEM_LABELS[system_key]
            ax_g = axes[r_idx, 2 * s_idx]
            ax_a = axes[r_idx, 2 * s_idx + 1]
            curves = all_curves.get(system_key, {}).get(regime, {})
            for name, style in CURVE_STYLE.items():
                d = curves.get(name)
                if d is None:
                    continue
                ax_g.plot(d["r"], d["g"], **style)
                ax_a.plot(d["th"], d["ad"], **style)
            ax_g.set_xlim(0, PDF_R_MAX)
            ax_a.set_xlim(0, 180)
            if r_idx == 0:
                ax_g.set_title(f"{label} — g(r)", fontsize=12)
                ax_a.set_title(f"{label} — ADF", fontsize=12)
            if s_idx == 0:
                ax_g.set_ylabel(regime, fontsize=11, rotation=0,
                                ha="right", va="center")
            if r_idx == n_rows - 1:
                ax_g.set_xlabel("r (Å)", fontsize=10)
                ax_a.set_xlabel("θ (deg)", fontsize=10)
            ax_g.grid(alpha=0.3, lw=0.4)
            ax_a.grid(alpha=0.3, lw=0.4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=s["color"], lw=s["lw"], ls=s["ls"])
               for s in CURVE_STYLE.values()]
    labels = [s["label"] for s in CURVE_STYLE.values()]
    fig.legend(handles, labels, loc="lower center", ncol=len(CURVE_STYLE),
               bbox_to_anchor=(0.5, -0.01), fontsize=11)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CACHE_NPZ.is_file():
        sys.exit(f"missing cache: {CACHE_NPZ}")
    all_curves = load_cache(CACHE_NPZ)
    print(f"[load] systems in cache: "
          f"{sorted(all_curves.keys())}", flush=True)
    for k, v in all_curves.items():
        print(f"  {k}: regimes={sorted(v.keys())}")

    plot_init_overlap(all_curves,
                      OUT_DIR / f"scale_invariance_init_overlap{DTYPE_SUFFIX}.png")
    plot_per_system(all_curves, OUT_DIR)
    plot_per_regime(all_curves, OUT_DIR)
    plot_full_overlay(all_curves,
                      OUT_DIR / f"scale_invariance_all_overlay{DTYPE_SUFFIX}.png")


if __name__ == "__main__":
    main()
