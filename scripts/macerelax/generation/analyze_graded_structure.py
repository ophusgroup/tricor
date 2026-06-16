"""Diagnose the spatial disorder gradient produced by
generate_graded_structure.py.

Loads the trajectory NPZ and, for the initial pack / bond_relax-cleaned /
MACE-relaxed frames, slices the cell into bins along the long axis and
computes per-slab structure:

  * a g(r)-like pair-distance heatmap (long-axis bin × r) — a crystalline
    slab shows sharp, deep-minimum peaks; an amorphous slab shows broad
    peaks with a shallow first minimum.
  * a per-slab "peak contrast" order metric = g(first peak) / g(first
    minimum): high for ordered slabs, ~1-2 for disordered ones.

The money plot is whether the order metric still tracks the long-axis order
profile in the FINAL (MACE-relaxed) frame — i.e. the gradient survived the
relaxation rather than collapsing into a uniform glass (MACE_RELAX_PILOT.md
§2g).

    python scripts/macerelax/generation/analyze_graded_structure.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms
from ase.neighborlist import neighbor_list


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR     = Path("/home/ehrdt/tricor/mace/data/graded_v1")
SYSTEM_LABEL = "SiO2_quartz_graded"
TRAJ_NPZ     = DATA_DIR / f"{SYSTEM_LABEL}_trajectory.npz"

N_SLABS    = 24            # bins along the long axis
R_MAX      = 6.0           # Å — pair-distance cutoff for the per-slab g(r)
R_BINS     = 120           # radial bins
SCAN_STEPS = [15, 30, 45, 60]  # MACE/FIRE step counts to compare (sliced from the
                           # single saved trajectory — see generate script's
                           # N_STEPS note).  The frame nearest each step is used.
OUT_PNG = DATA_DIR / f"{SYSTEM_LABEL}_gradient_analysis.png"

# ══════════════════════════════════════════════════════════════════════════════


def per_slab_gr(numbers, positions, cell, long_axis, n_slabs, r_max, r_bins):
    """Return (slab_centers_frac, r_centers, gr[n_slabs, r_bins]).

    g(r)-like: per-slab pair-distance histogram normalised by the slab's
    center-atom count and the spherical-shell volume.  Density is constant
    across slabs by construction, so the curves are directly comparable.
    """
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    i, _, d = neighbor_list("ijd", atoms, r_max)

    L = float(cell[long_axis, long_axis])
    s_i = np.mod(positions[i, long_axis] / L, 1.0)        # slab of center atom
    slab_of_pair = np.clip((s_i * n_slabs).astype(int), 0, n_slabs - 1)

    s_atom = np.mod(positions[:, long_axis] / L, 1.0)
    slab_of_atom = np.clip((s_atom * n_slabs).astype(int), 0, n_slabs - 1)
    atoms_per_slab = np.bincount(slab_of_atom, minlength=n_slabs)

    r_edges = np.linspace(0.0, r_max, r_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    shell_vol = 4.0 / 3.0 * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

    gr = np.zeros((n_slabs, r_bins), dtype=np.float64)
    for sb in range(n_slabs):
        sel = slab_of_pair == sb
        if not np.any(sel) or atoms_per_slab[sb] == 0:
            continue
        hist, _ = np.histogram(d[sel], bins=r_edges)
        gr[sb] = hist / (atoms_per_slab[sb] * shell_vol)

    slab_centers = (np.arange(n_slabs) + 0.5) / n_slabs
    return slab_centers, r_centers, gr


def peak_contrast(r_centers, gr_row):
    """g(first peak) / g(first minimum) — ordered slabs score high."""
    if not np.any(gr_row > 0):
        return np.nan
    pk_i = int(np.argmax(gr_row))
    if pk_i >= len(gr_row) - 2:
        return np.nan
    tail = gr_row[pk_i:]
    mn_rel = int(np.argmin(tail))
    g_peak = gr_row[pk_i]
    g_min = tail[mn_rel]
    return float(g_peak / max(g_min, 1e-6))


def main() -> None:
    if not TRAJ_NPZ.is_file():
        raise SystemExit(
            f"trajectory not found: {TRAJ_NPZ}\n"
            f"run generate_graded_structure.py first."
        )
    z = np.load(TRAJ_NPZ, allow_pickle=True)
    numbers = z["species_numbers"]
    cell = z["cell"].astype(np.float64)
    long_axis = int(z["long_axis"])
    order = z["order"]                       # per-atom target order (initial frame)
    traj = z["positions"].astype(np.float64)         # (S, N, 3) every saved frame
    snap_steps = z["snapshot_steps"].astype(int)     # (S,) step index per frame
    max_step = int(snap_steps.max())

    def frame_at_step(step):
        """Trajectory frame nearest the requested FIRE step (clamped)."""
        target = min(step, max_step)
        return traj[int(np.argmin(np.abs(snap_steps - target)))]

    # Baseline = bond_relax-cleaned (pre-MACE) + one frame per scan step, so the
    # bottom row shows how the spatial gradient evolves with MACE step count.
    frames = {"cleaned (0)": z["cleaned_positions"].astype(np.float64)}
    for st in SCAN_STEPS:
        frames[f"step {min(st, max_step)}"] = frame_at_step(st)

    fig, axes = plt.subplots(2, len(frames), figsize=(4.2 * len(frames), 8),
                             constrained_layout=True)

    contrasts = {}
    for col, (name, pos) in enumerate(frames.items()):
        slab_c, r_c, gr = per_slab_gr(
            numbers, pos, cell, long_axis, N_SLABS, R_MAX, R_BINS)
        contrasts[name] = np.array([peak_contrast(r_c, gr[s])
                                    for s in range(N_SLABS)])

        ax = axes[0, col]
        im = ax.imshow(gr, aspect="auto", origin="lower",
                       extent=[r_c[0], r_c[-1], 0.0, 1.0], cmap="viridis")
        ax.set_title(f"{name}\nper-slab g(r)")
        ax.set_xlabel("r (Å)")
        if col == 0:
            ax.set_ylabel("fractional position along long axis")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Bottom row: order metric + the target order profile, shared across frames.
    # Build the target straight from the stored per-atom `order` (works for any
    # profile), binned by initial-frame slab membership.
    init_pos = z["initial_positions"].astype(np.float64)
    L = float(cell[long_axis, long_axis])
    slab_init = np.clip(
        (np.mod(init_pos[:, long_axis] / L, 1.0) * N_SLABS).astype(int),
        0, N_SLABS - 1,
    )
    target_order_by_slab = np.array([
        order[slab_init == sb].mean() if np.any(slab_init == sb) else np.nan
        for sb in range(N_SLABS)
    ])

    for col, name in enumerate(frames):
        ax = axes[1, col]
        c = contrasts[name]
        ax.plot(np.linspace(0, 1, N_SLABS), c, "-o", ms=3, label="peak contrast")
        ax.set_xlabel("fractional position along long axis")
        if col == 0:
            ax.set_ylabel("peak contrast (order →)")
        ax2 = ax.twinx()
        ax2.plot(np.linspace(0, 1, N_SLABS), target_order_by_slab,
                 "r--", alpha=0.6, label="target order")
        ax2.set_ylim(-0.05, 1.05)
        if col == len(frames) - 1:
            ax2.set_ylabel("target order profile", color="r")
        ax.set_title(f"{name}: order vs position")

    fig.suptitle(
        f"{SYSTEM_LABEL}: spatial disorder gradient vs MACE step count "
        f"(per-slab order should track the red target; watch it fade as steps grow)",
        fontsize=12,
    )
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")

    # Numeric readout: gradient retention (order-vs-target correlation) for every
    # frame.  A correlation that stays high at step 10/20 but drops by step 30
    # tells you where the gradient starts collapsing into a uniform glass (§2g).
    print("\ngradient retention (peak-contrast vs target order, +1 = preserved):")
    for name in frames:
        c = contrasts[name]
        good = np.isfinite(c) & np.isfinite(target_order_by_slab)
        corr = (np.corrcoef(c[good], target_order_by_slab[good])[0, 1]
                if good.sum() > 3 else float("nan"))
        print(f"  {name:>12s}: {corr:+.3f}")


if __name__ == "__main__":
    main()
