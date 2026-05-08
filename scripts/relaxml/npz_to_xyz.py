"""Convert a tricor surrogate .npz trajectory to extended XYZ for OVITO.

Edit the CONFIG block below, then run:
    python npz_to_xyz.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

# Path to the input .npz file.
INPUT = "./data/si_test_cells/cell080/si_MRO_cell080_idx00007_seed000300007.npz"

# Output .xyz path.  None = auto: <input>.xyz next to input.
OUTPUT = None

# What to write:
#   "trajectory" — full positions (S, N, 3) as a scrubbable animation in OVITO
#   "best"       — single frame, best_positions (loss minimum from shell_relax)
#   "final"      — single frame, final_positions (last step)
MODE = "final"

# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def _atoms_from(positions, numbers, cell):
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def main() -> None:
    in_path = Path(INPUT).resolve()
    out_path = Path(OUTPUT).resolve() if OUTPUT else in_path.with_suffix(".xyz")

    with np.load(in_path) as npz:
        cell = np.asarray(npz["cell"], dtype=np.float64)
        numbers = np.asarray(npz["species_numbers"], dtype=np.int64)
        if MODE == "best":
            frames = [_atoms_from(np.asarray(npz["best_positions"]), numbers, cell)]
        elif MODE == "final":
            frames = [_atoms_from(np.asarray(npz["final_positions"]), numbers, cell)]
        elif MODE == "trajectory":
            traj = np.asarray(npz["positions"])
            frames = [_atoms_from(traj[i], numbers, cell) for i in range(traj.shape[0])]
        else:
            raise ValueError(f"MODE must be 'trajectory', 'best', or 'final'; got {MODE!r}")

    write(out_path, frames, format="extxyz")
    print(f"Wrote {len(frames)} frame(s) to {out_path}")


if __name__ == "__main__":
    main()
