"""Generate structures using a trained flow matching model.

Edit the CONFIG section below, then run:
    python generate_flowmatch.py
"""

from pathlib import Path

import torch
import numpy as np
import ase.io

from tricor.flowmatch import LitFlowMatch, generate, positions_to_atoms
from tricor.differentiable_pdf import DifferentiablePDFADF

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Trained model
CHECKPOINT = "./lightning_logs/flowmatch-si3n4/version_0/checkpoints/last.ckpt"
SPECIES = [7, 14]                        # must match training species

# Target (provide ONE of these)
TARGET_STRUCTURE = "/path/to/reference.xyz"   # compute targets from this
TARGET_G2_FILE = None                         # OR load pre-computed
TARGET_ADF_FILE = None

# Generation cell
NUM_ATOMS = 400
CELL_SIZE = 25.0                         # cubic cell side length (A)
ATOM_FRACTIONS = [3/7, 4/7]             # fraction per species (Si3N4: 3/7 N, 4/7 Si)

# Spectral settings (must match training)
R_MAX = 10.0
R_STEP = 0.05
PHI_NUM_BINS = 90
SIGMA_R = 0.15
SIGMA_PHI = 0.1

# ODE integration
NUM_STEPS = 30                           # 20-50 is typical
METHOD = "midpoint"                      # "euler" or "midpoint"
CUTOFF = 5.0

# Output
NUM_RUNS = 10
OUTPUT_DIR = "./generated_flowmatch/"
SEED = 42
DEVICE = "cuda"

# ══════════════════════════════════════════════════════════════════════════════


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {CHECKPOINT}")
    lit = LitFlowMatch.load_from_checkpoint(CHECKPOINT)
    lit.ema_model.to(device)
    lit.ema_model.eval()

    num_species = len(SPECIES)

    # Compute or load targets
    if TARGET_STRUCTURE is not None:
        print(f"Computing targets from: {TARGET_STRUCTURE}")
        calc = DifferentiablePDFADF(
            r_max=R_MAX, r_step=R_STEP, phi_num_bins=PHI_NUM_BINS,
            sigma_r=SIGMA_R, sigma_phi=SIGMA_PHI, species=SPECIES,
        ).double()
        ref = ase.io.read(TARGET_STRUCTURE)
        pos_ref = torch.tensor(ref.positions, dtype=torch.float64)
        sp_ref = torch.tensor(ref.numbers, dtype=torch.long)
        cell_ref = torch.tensor(ref.cell.array, dtype=torch.float64)
        with torch.no_grad():
            g2_target, adf_target = calc.compute(pos_ref, sp_ref, cell_ref)
        # Normalize by atom count (same as training)
        g2_target = (g2_target / len(ref)).float().unsqueeze(0).to(device)
        adf_target = (adf_target / len(ref)).float().unsqueeze(0).to(device)
    elif TARGET_G2_FILE is not None and TARGET_ADF_FILE is not None:
        g2_target = torch.load(TARGET_G2_FILE).unsqueeze(0).to(device)
        adf_target = torch.load(TARGET_ADF_FILE).unsqueeze(0).to(device)
    else:
        raise ValueError("Set either TARGET_STRUCTURE or both TARGET_G2_FILE and TARGET_ADF_FILE")

    # Species assignment
    fracs = np.array(ATOM_FRACTIONS)
    fracs = fracs / fracs.sum()
    counts = np.round(fracs * NUM_ATOMS).astype(int)
    counts[-1] = NUM_ATOMS - counts[:-1].sum()

    species_list = []
    z_rows = []
    for i, (Z, count) in enumerate(zip(SPECIES, counts)):
        species_list.extend([Z] * count)
        onehot = torch.zeros(count, num_species)
        onehot[:, i] = 1.0
        z_rows.append(onehot)

    species = torch.tensor(species_list, dtype=torch.long, device=device)
    z = torch.cat(z_rows, dim=0).to(device)
    cell = torch.diag(torch.tensor([CELL_SIZE] * 3, device=device))
    comp_frac = torch.tensor([fracs], dtype=torch.float32, device=device)

    print(f"Generating: {NUM_ATOMS} atoms in {CELL_SIZE} A cell")
    print(f"  Species counts: {dict(zip(SPECIES, counts.tolist()))}")
    print(f"  ODE: {NUM_STEPS} {METHOD} steps")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for run in range(NUM_RUNS):
        pos = generate(
            cell=cell, num_atoms=NUM_ATOMS, z=z,
            velocity_model=lit.ema_model,
            g2_target=g2_target, adf_target=adf_target,
            comp_frac=comp_frac,
            cutoff=CUTOFF, num_steps=NUM_STEPS, method=METHOD,
        )

        atoms = positions_to_atoms(pos, cell, species)
        outfile = output_dir / f"generated_{run:03d}.xyz"
        ase.io.write(str(outfile), atoms)
        print(f"  Run {run+1}/{NUM_RUNS}: saved {outfile}")

    print(f"\nDone. {NUM_RUNS} structures saved to {output_dir}")


if __name__ == "__main__":
    main()
