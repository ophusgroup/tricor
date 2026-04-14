"""Generate structures using unconditional flow matching + spectral guidance.

Like GLASS's conditional generation but with flow matching for speed.
The model is unconditional; the target g2/ADF steers generation via
gradient guidance at inference time.

Edit the CONFIG section below, then run:
    python generate_flowmatch_guided.py
"""

from pathlib import Path

import torch
import numpy as np
import ase.io

from tricor.flowmatch.velocity_model_uncond import LitUncondFlowMatch
from tricor.flowmatch.sampler_guided import (
    generate_unconditional,
    generate_guided,
    positions_to_atoms,
)
from tricor.differentiable_pdf import DifferentiablePDFADF, DifferentiableSpectralLoss

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Trained model
CHECKPOINT = "./lightning_logs/flowmatch-uncond-si3n4/version_0/checkpoints/last.ckpt"
SPECIES = [7, 14]

# Target structure (compute g2/ADF from this)
TARGET_STRUCTURE = "/pscratch/sd/e/ehrdt/mcstructgen/smallcell/Si3N4(10)_d90_g3_jit2_s0.15_m2.xyz"

# Generation cell
NUM_ATOMS = None                         # if None, match reference
CELL_SIZE = 25.0
ATOM_FRACTIONS = [3/7, 4/7]             # N, Si for Si3N4

# Spectral calculator
R_MAX = 10.0
R_STEP = 0.05
PHI_NUM_BINS = 90
SIGMA_R = 0.15
SIGMA_PHI = 0.1
PDF_WEIGHT = 1.0
ADF_WEIGHT = 1.0

# Guidance
W = 3000.0                               # guidance weight
NUM_STEPS = 50                           # ODE steps with guidance
CUTOFF = 5.0

# Output
NUM_RUNS = 5
OUTPUT_DIR = "./generated_guided/"
SEED = 42
DEVICE = "cuda"

# ══════════════════════════════════════════════════════════════════════════════


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading: {CHECKPOINT}")
    lit = LitUncondFlowMatch.load_from_checkpoint(CHECKPOINT, map_location=device)
    lit.ema_model.to(device)
    lit.ema_model.eval()

    num_species = len(SPECIES)

    # Spectral calculator and loss
    calc = DifferentiablePDFADF(
        r_max=R_MAX, r_step=R_STEP, phi_num_bins=PHI_NUM_BINS,
        sigma_r=SIGMA_R, sigma_phi=SIGMA_PHI, species=SPECIES,
    ).double().to(device)

    loss_fn = DifferentiableSpectralLoss(calc, pdf_weight=PDF_WEIGHT, adf_weight=ADF_WEIGHT)

    # Compute targets from reference
    print(f"Computing targets from: {TARGET_STRUCTURE}")
    ref = ase.io.read(TARGET_STRUCTURE)
    ref_pos = torch.tensor(ref.positions, dtype=torch.float64, device=device)
    ref_sp = torch.tensor(ref.numbers, dtype=torch.long, device=device)
    ref_cell = torch.tensor(ref.cell.array, dtype=torch.float64, device=device)
    with torch.no_grad():
        target_g2, target_adf = calc.compute(ref_pos, ref_sp, ref_cell)

    num_atoms = NUM_ATOMS if NUM_ATOMS is not None else len(ref)

    # Species assignment
    fracs = np.array(ATOM_FRACTIONS)
    fracs = fracs / fracs.sum()
    counts = np.round(fracs * num_atoms).astype(int)
    counts[-1] = num_atoms - counts[:-1].sum()

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

    print(f"Generating {NUM_RUNS} structures: {num_atoms} atoms, {CELL_SIZE} A cell")
    print(f"  Species counts: {dict(zip(SPECIES, counts.tolist()))}")
    print(f"  Guidance: w={W}, {NUM_STEPS} steps")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
        pos = generate_guided(
            cell=cell, num_atoms=num_atoms, z=z,
            velocity_model=lit.ema_model,
            spectral_loss_fn=loss_fn,
            target_g2=target_g2, target_adf=target_adf,
            species=species,
            cutoff=CUTOFF, w=W, num_steps=NUM_STEPS,
            verbose=True,
        )

        atoms = positions_to_atoms(pos, cell, species)
        outfile = output_dir / f"generated_{run:03d}.xyz"
        ase.io.write(str(outfile), atoms)
        print(f"  Saved: {outfile}")

    print(f"\nDone. {NUM_RUNS} structures saved to {output_dir}")


if __name__ == "__main__":
    main()
