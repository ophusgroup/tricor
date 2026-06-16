"""Re-emit the regime-overlay plot at an arbitrary frame index from the
existing v3_fire sweep trajectories on disk.

Uses the GPU PDF/ADF module from the sweep's evolution-plot path so the
8-regime compute completes in seconds rather than minutes.

Run:
    /home/ehrdt/miniforge3/envs/mace/bin/python scratch/replot_sweep_step.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/ehrdt/tricor/mace")
from ase.io import read   # noqa: E402

from sweep_regimes_sio2 import (   # noqa: E402
    _build_pdf_module, _compute_pdf_adf_with,
    _make_overlay, _peak_metrics,
    REGIMES, OUT_DIR, traj_path,
    grids,
)

# === CONFIG ============================================================
FRAME_INDEX = 80
SPECIES = [8, 14]                  # SiO2: O (Z=8) then Si (Z=14)
OUT_PNG = OUT_DIR / f"sio2_regime_compare_mace_v3_fire_step{FRAME_INDEX}.png"
# =======================================================================


def _gather_snaps_gpu(species, frame_index, mod, device):
    """GPU-accelerated mirror of sweep_regimes_sio2._gather_snaps."""
    r_grid, phi_grid = grids()
    snaps: dict = {}
    for regime in REGIMES:
        p = traj_path(regime)
        if not p.is_file() or p.stat().st_size == 0:
            print(f"  [{regime}] missing trajectory — skipped")
            continue
        atoms = read(p, index=frame_index)
        g2, adf, z, V = _compute_pdf_adf_with(mod, device, atoms, species)
        snaps[regime] = {"g2": g2, "adf": adf, "z": z, "V": V,
                          "n_atoms": int(len(atoms))}
        print(f"  [{regime}] PDF/ADF computed  N={len(atoms)}")
    return snaps, r_grid, phi_grid


def main() -> None:
    print(f"Frame {FRAME_INDEX} overlay (GPU PDF/ADF)...")
    mod, device = _build_pdf_module(SPECIES)
    print(f"  device: {device}")

    snaps, r_grid, phi_grid = _gather_snaps_gpu(
        SPECIES, FRAME_INDEX, mod, device,
    )
    if not snaps:
        sys.exit(f"No trajectories found under {OUT_DIR}")

    _make_overlay(
        snaps, r_grid, phi_grid, SPECIES,
        f"step {FRAME_INDEX} (FIRE maxstep=0.3, CELL=50)", OUT_PNG,
    )
    print(f"Saved: {OUT_PNG}")

    rows = _peak_metrics(snaps, r_grid, phi_grid, SPECIES,
                          tag=f"step{FRAME_INDEX}")
    if not rows:
        return
    print(f"\nFirst-peak diagnostic at step {FRAME_INDEX}:")
    print(f"  {'regime':<18s} {'panel':<18s} {'x':>8s} {'y':>8s} {'fwhm':>8s}")
    for r in rows:
        x = r.get("x"); y = r.get("y"); fw = r.get("fwhm")
        xs = f"{x:>8.3f}" if isinstance(x, (int, float)) else "       —"
        ys = f"{y:>8.3f}" if isinstance(y, (int, float)) else "       —"
        fws = f"{fw:>8.3f}" if isinstance(fw, (int, float)) else "       —"
        print(f"  {r['regime']:<18s} {r['panel']:<18s} {xs} {ys} {fws}")


if __name__ == "__main__":
    main()
