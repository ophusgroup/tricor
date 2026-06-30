"""Generate structures using a trained GLASS score model with PDF/ADF guidance.

Edit the CONFIG section below, then run:
    python generate_structures.py
"""

from pathlib import Path

import torch
import numpy as np
import ase.io

from tricor.glass import LitScoreNet
from tricor.glass.sampler import generate, positions_to_atoms
from tricor.differentiable_pdf import DifferentiablePDFADF, DifferentiableSpectralLoss

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

# Trained model
CHECKPOINT = "./lightning_logs/glass-si3n4/version_0/checkpoints/last.ckpt"
SPECIES = [7, 14]                        # must match training species

# Target (provide ONE of these)
TARGET_STRUCTURE = "/path/to/reference.xyz"   # compute targets from this structure
TARGET_G2_FILE = None                         # OR load pre-computed targets
TARGET_ADF_FILE = None

# Generation cell
NUM_ATOMS = 400
CELL_SIZE = 25.0                         # cubic cell side length (A)
ATOM_FRACTIONS = [3/7, 4/7]             # fraction per species (Si3N4: 3/7 N, 4/7 Si)

# Spectral calculator settings
R_MAX = 10.0                             # PDF cutoff (A)
R_STEP = 0.05                            # radial bin width (A)
PHI_NUM_BINS = 90                        # angular bins
SIGMA_R = 0.15                           # PDF Gaussian bandwidth (A)
SIGMA_PHI = 0.1                          # ADF Gaussian bandwidth (rad)
PDF_WEIGHT = 1.0                         # weight for PDF loss
ADF_WEIGHT = 1.0                         # weight for ADF loss

# Denoising parameters
W = 3000.0                               # guidance weight
UNCOND_STEPS = 512                       # unconditional stage steps
COND_STEPS = 512                         # conditional stage steps
CUTOFF = 5.0                             # graph construction cutoff (A)
MIN_DIST = None                          # minimum-distance veto threshold (A), or None

# Output
NUM_RUNS = 10                            # independent generation runs
OUTPUT_DIR = "./generated/"
SEED = 42
DEVICE = "cuda"                          # "cuda" or "cpu"

# ══════════════════════════════════════════════════════════════════════════════


def compute_targets_from_structure(atoms, calc):
    positions = torch.tensor(atoms.positions, dtype=torch.float64)
    species = torch.tensor(atoms.numbers, dtype=torch.long)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64)
    with torch.no_grad():
        g2, adf = calc.compute(positions, species, cell)
    return g2, adf


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {CHECKPOINT}")
    score_net = LitScoreNet.load_from_checkpoint(CHECKPOINT, map_location=device)
    score_net.ema_model.to(device)
    score_net.ema_model.eval()

    num_species = len(SPECIES)

    # Spectral calculator
    calc = DifferentiablePDFADF(
        r_max=R_MAX, r_step=R_STEP, phi_num_bins=PHI_NUM_BINS,
        sigma_r=SIGMA_R, sigma_phi=SIGMA_PHI, species=SPECIES,
    ).double().to(device)

    loss_fn = DifferentiableSpectralLoss(calc, pdf_weight=PDF_WEIGHT, adf_weight=ADF_WEIGHT)

    # Targets
    if TARGET_STRUCTURE is not None:
        print(f"Computing targets from: {TARGET_STRUCTURE}")
        ref_atoms = ase.io.read(TARGET_STRUCTURE)
        target_g2, target_adf = compute_targets_from_structure(ref_atoms, calc)
    elif TARGET_G2_FILE is not None and TARGET_ADF_FILE is not None:
        target_g2 = torch.load(TARGET_G2_FILE, map_location=device)
        target_adf = torch.load(TARGET_ADF_FILE, map_location=device)
    else:
        raise ValueError("Set either TARGET_STRUCTURE or both TARGET_G2_FILE and TARGET_ADF_FILE")

    target_g2 = target_g2.to(device)
    target_adf = target_adf.to(device)

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

    print(f"Generating: {NUM_ATOMS} atoms in {CELL_SIZE} A cell")
    print(f"  Species counts: {dict(zip(SPECIES, counts.tolist()))}")
    print(f"  w={W}, uncond_steps={UNCOND_STEPS}, cond_steps={COND_STEPS}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run + 1}/{NUM_RUNS} ---")
        pos = generate(
            cell=cell, num_atoms=NUM_ATOMS, z=z,
            score_model=score_net.ema_model,
            spectral_loss_fn=loss_fn,
            target_g2=target_g2, target_adf=target_adf, species=species,
            cutoff=CUTOFF, w=W,
            uncond_steps=UNCOND_STEPS, cond_steps=COND_STEPS,
            min_dist_threshold=MIN_DIST,
            verbose=True,
        )

        atoms = positions_to_atoms(pos, cell, species)
        outfile = output_dir / f"generated_{run:03d}.xyz"
        ase.io.write(str(outfile), atoms)
        print(f"  Saved: {outfile} ({len(atoms)} atoms)")

    print(f"\nDone. {NUM_RUNS} structures saved to {output_dir}")


if __name__ == "__main__":
    main()
