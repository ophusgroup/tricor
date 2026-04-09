"""Pre-compute PDF/ADF targets from a reference structure.

Saves g2 and ADF tensors as .pt files that can be loaded by
generate_structures.py via TARGET_G2_FILE and TARGET_ADF_FILE.

Edit the CONFIG section below, then run:
    python compute_targets.py
"""

from pathlib import Path

import torch
import ase.io

from tricor.differentiable_pdf import DifferentiablePDFADF

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

STRUCTURE = "/path/to/reference.xyz"
SPECIES = [7, 14]
R_MAX = 10.0
R_STEP = 0.05
PHI_NUM_BINS = 90
SIGMA_R = 0.15
SIGMA_PHI = 0.1
OUTPUT_PREFIX = "./targets/si3n4_ref"    # produces {prefix}_g2.pt and {prefix}_adf.pt

# ══════════════════════════════════════════════════════════════════════════════


def main():
    atoms = ase.io.read(STRUCTURE)
    print(f"Structure: {len(atoms)} atoms, cell={atoms.cell.lengths()}")
    print(f"Species: {sorted(set(atoms.numbers.tolist()))}")

    calc = DifferentiablePDFADF(
        r_max=R_MAX, r_step=R_STEP, phi_num_bins=PHI_NUM_BINS,
        sigma_r=SIGMA_R, sigma_phi=SIGMA_PHI, species=SPECIES,
    ).double()

    positions = torch.tensor(atoms.positions, dtype=torch.float64)
    species = torch.tensor(atoms.numbers, dtype=torch.long)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64)

    print("Computing g2 and ADF...")
    with torch.no_grad():
        g2, adf = calc.compute(positions, species, cell)

    print(f"  g2 shape: {g2.shape}")
    print(f"  ADF shape: {adf.shape}")
    print(f"  Pair labels: {calc.pair_labels}")
    print(f"  Triplet labels: {calc.triplet_labels}")

    output_dir = Path(OUTPUT_PREFIX).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    g2_path = f"{OUTPUT_PREFIX}_g2.pt"
    adf_path = f"{OUTPUT_PREFIX}_adf.pt"
    torch.save(g2, g2_path)
    torch.save(adf, adf_path)
    print(f"Saved: {g2_path}")
    print(f"Saved: {adf_path}")


if __name__ == "__main__":
    main()
