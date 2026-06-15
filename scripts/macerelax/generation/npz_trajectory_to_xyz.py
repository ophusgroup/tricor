"""Convert a graded-structure trajectory NPZ into a multi-frame extxyz that
OVITO can open as an animation.

Each saved FIRE frame becomes one extxyz frame, carrying the per-atom `order`
coordinate (0=amorphous → 1=crystalline) as a column so you can colour the
relaxation by initial disorder.  The frame's `step` (FIRE step index) is
written into the per-frame comment line.

    python scripts/macerelax/generation/npz_trajectory_to_xyz.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR     = Path("/home/ehrdt/tricor/mace/data/graded_v1")
SYSTEM_LABEL = "SiO2_quartz_graded"
TRAJ_NPZ     = DATA_DIR / f"{SYSTEM_LABEL}_trajectory.npz"
OUT_XYZ      = DATA_DIR / f"{SYSTEM_LABEL}_trajectory.xyz"   # multi-frame extxyz

# ══════════════════════════════════════════════════════════════════════════════


def npz_to_trajectory_xyz(traj_npz: Path, out_xyz: Path) -> int:
    """Write every frame in `traj_npz` to a multi-frame extxyz. Returns n_frames."""
    z = np.load(traj_npz, allow_pickle=True)
    positions = z["positions"].astype(np.float64)        # (S, N, 3)
    numbers = z["species_numbers"]
    cell = z["cell"].astype(np.float64)
    order = z["order"].astype(np.float64) if "order" in z else None
    steps = (z["snapshot_steps"].astype(int) if "snapshot_steps" in z
             else np.arange(len(positions)))

    frames = []
    for i in range(len(positions)):
        a = Atoms(numbers=numbers, positions=positions[i], cell=cell, pbc=True)
        if order is not None:
            a.set_array("order", order)
        a.info["step"] = int(steps[i])
        frames.append(a)

    out_xyz.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(out_xyz), frames, format="extxyz")
    return len(frames)


def main() -> None:
    if not TRAJ_NPZ.is_file():
        raise SystemExit(f"trajectory not found: {TRAJ_NPZ}")
    n = npz_to_trajectory_xyz(TRAJ_NPZ, OUT_XYZ)
    print(f"wrote {n} frames to {OUT_XYZ}")
    print("open in OVITO; colour by the 'order' particle property "
          "(0=amorphous → 1=crystalline).")


if __name__ == "__main__":
    main()
