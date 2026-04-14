"""Precompute g2/ADF labels for all structures in a directory.

Saves a single .pt file containing g2, ADF, and composition fractions
for every structure. The flow matching dataset can then load this
cache instead of recomputing.

Edit the CONFIG section below, then run:
    python precompute_labels.py
"""

import time
from pathlib import Path

import torch
import numpy as np
import ase.io

from tricor.differentiable_pdf import DifferentiablePDFADF

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = "/pscratch/sd/e/ehrdt/mcstructgen/smallcell/"
SPECIES = [7, 14]
R_MAX = 10.0
R_STEP = 0.05
PHI_NUM_BINS = 90
SIGMA_R = 0.15
SIGMA_PHI = 0.1
OUTPUT_FILE = "/pscratch/sd/e/ehrdt/mcstructgen/smallcell_labels.pt"

# ══════════════════════════════════════════════════════════════════════════════


def main():
    data_dir = Path(DATA_DIR)
    files = sorted(
        list(data_dir.glob("*.extxyz"))
        + list(data_dir.glob("*.vasp"))
        + list(data_dir.glob("*.cif"))
        + list(data_dir.glob("*.xyz"))
    )
    print(f"Found {len(files)} structure files in {data_dir}")

    calc = DifferentiablePDFADF(
        r_max=R_MAX, r_step=R_STEP, phi_num_bins=PHI_NUM_BINS,
        sigma_r=SIGMA_R, sigma_phi=SIGMA_PHI, species=SPECIES,
    ).double()

    num_species = len(SPECIES)
    results = []

    for i, f in enumerate(files):
        t0 = time.time()
        atoms = ase.io.read(f)
        atoms.wrap()
        n = len(atoms)

        positions = torch.tensor(atoms.positions, dtype=torch.float64)
        sp = torch.tensor(atoms.numbers, dtype=torch.long)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float64)

        with torch.no_grad():
            g2, adf = calc.compute(positions, sp, cell)

        # Normalize by atom count
        g2 = (g2 / max(n, 1)).float()
        adf = (adf / max(n, 1)).float()

        # Composition fractions
        comp_frac = np.zeros(num_species, dtype=np.float32)
        for j, Z in enumerate(SPECIES):
            comp_frac[j] = (atoms.numbers == Z).sum() / n

        results.append({
            "filename": f.name,
            "g2": g2,
            "adf": adf,
            "comp_frac": torch.tensor(comp_frac),
            "num_atoms": n,
            "cell": torch.tensor(atoms.cell.array, dtype=torch.float32),
            "positions": torch.tensor(atoms.positions, dtype=torch.float32),
            "numbers": torch.tensor(atoms.numbers, dtype=torch.long),
        })

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(files)}] {f.name}: {n} atoms, {elapsed:.1f}s")

    torch.save(results, OUTPUT_FILE)
    print(f"\nSaved {len(results)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
