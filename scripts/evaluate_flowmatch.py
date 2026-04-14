"""Evaluate a trained flow matching model by generating structures
and comparing their g2/ADF against a reference.

All comparisons use properly normalized functions:
  - g(r): normalized by ideal gas density so it approaches 1 at large r
  - ADF: normalized to a probability density (integral = 1)

Edit the CONFIG section below, then run:
    python evaluate_flowmatch.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import ase.io
import math

from tricor.flowmatch import LitFlowMatch, generate, positions_to_atoms
from tricor.differentiable_pdf import DifferentiablePDFADF

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT = "./lightning_logs/flowmatch-si3n4/version_3/checkpoints/last.ckpt"
REFERENCE_STRUCTURE = "/pscratch/sd/e/ehrdt/mcstructgen/smallcell/Si3N4(10)_d90_g3_jit2_s0.15_m2.xyz"
SPECIES = [7, 14]
ATOM_FRACTIONS = [3/7, 4/7]             # N, Si for Si3N4

# Generation settings
CELL_SIZE = 25.0
NUM_ATOMS = None                         # if None, match the reference
NUM_STEPS = 30
METHOD = "midpoint"
CUTOFF = 5.0
NUM_RUNS = 3

# Spectral settings (must match training)
R_MAX = 10.0
R_STEP = 0.05
PHI_NUM_BINS = 90
SIGMA_R = 0.15
SIGMA_PHI = 0.1

DEVICE = "cuda"
SEED = 42
SAVE_STRUCTURES = True
OUTPUT_DIR = "./eval_output/"

# ══════════════════════════════════════════════════════════════════════════════


def normalize_g2(g2_raw, r, num_center, num_neighbor, volume, dr):
    """Normalize raw g2 counts to standard g(r) that approaches 1 at large r.

    g(r) = raw_counts / (N_center * rho_neighbor * 4*pi*r^2 * dr)
    """
    rho = num_neighbor / volume
    shell_vol = 4.0 * math.pi * r ** 2 * dr
    denom = num_center * rho * shell_vol
    return g2_raw / np.maximum(denom, 1e-12)


def normalize_adf(adf_raw, dphi):
    """Normalize raw ADF counts to probability density (integral ≈ 1)."""
    total = adf_raw.sum() * dphi
    if total > 1e-12:
        return adf_raw / total
    return adf_raw


def compute_spectra(atoms, calc, species_list):
    """Compute normalized g2 and ADF for an ASE Atoms object."""
    pos = torch.tensor(atoms.positions, dtype=torch.float64)
    sp = torch.tensor(atoms.numbers, dtype=torch.long)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64)

    with torch.no_grad():
        g2_raw, adf_raw = calc.compute(pos, sp, cell)

    g2_raw = g2_raw.numpy()
    adf_raw = adf_raw.numpy()

    r = calc.r_grid.numpy()
    dr = float(calc.r_step)
    volume = abs(np.linalg.det(atoms.cell.array))
    num_species = len(species_list)

    # Normalize each g2 channel
    g2_norm = np.zeros_like(g2_raw)
    for i in range(num_species):
        Z_i = species_list[i]
        n_i = (atoms.numbers == Z_i).sum()
        for j in range(num_species):
            Z_j = species_list[j]
            n_j = (atoms.numbers == Z_j).sum()
            g2_norm[i, j] = normalize_g2(g2_raw[i, j], r, n_i, n_j, volume, dr)

    # Normalize each ADF channel
    phi = calc.phi_grid.numpy()
    dphi = phi[1] - phi[0] if len(phi) > 1 else 1.0
    adf_norm = np.zeros_like(adf_raw)
    for t_idx in range(adf_raw.shape[0]):
        adf_norm[t_idx] = normalize_adf(adf_raw[t_idx], dphi)

    return g2_norm, adf_norm


def main():
    torch.manual_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load reference
    ref = ase.io.read(REFERENCE_STRUCTURE)
    print(f"Reference: {len(ref)} atoms, cell={ref.cell.lengths()}")

    num_atoms = NUM_ATOMS if NUM_ATOMS is not None else len(ref)
    num_species = len(SPECIES)

    # Spectral calculator
    calc = DifferentiablePDFADF(
        r_max=R_MAX, r_step=R_STEP, phi_num_bins=PHI_NUM_BINS,
        sigma_r=SIGMA_R, sigma_phi=SIGMA_PHI, species=SPECIES,
    ).double()

    # Normalized reference spectra
    print("Computing reference spectra...")
    ref_g2, ref_adf = compute_spectra(ref, calc, SPECIES)

    # Target for the model (normalized by atom count, matching training convention)
    ref_pos = torch.tensor(ref.positions, dtype=torch.float64)
    ref_sp = torch.tensor(ref.numbers, dtype=torch.long)
    ref_cell = torch.tensor(ref.cell.array, dtype=torch.float64)
    with torch.no_grad():
        raw_g2, raw_adf = calc.compute(ref_pos, ref_sp, ref_cell)
    target_g2 = (raw_g2 / len(ref)).float().unsqueeze(0).to(device)
    target_adf = (raw_adf / len(ref)).float().unsqueeze(0).to(device)

    # Load model
    print(f"Loading: {CHECKPOINT}")
    lit = LitFlowMatch.load_from_checkpoint(CHECKPOINT, map_location=device)
    lit.ema_model.to(device)
    lit.ema_model.eval()

    # Species assignment
    fracs = np.array(ATOM_FRACTIONS)
    fracs = fracs / fracs.sum()
    counts = np.round(fracs * num_atoms).astype(int)
    counts[-1] = num_atoms - counts[:-1].sum()

    species_list_gen = []
    z_rows = []
    for i, (Z, count) in enumerate(zip(SPECIES, counts)):
        species_list_gen.extend([Z] * count)
        onehot = torch.zeros(count, num_species)
        onehot[:, i] = 1.0
        z_rows.append(onehot)

    species_gen = torch.tensor(species_list_gen, dtype=torch.long, device=device)
    z = torch.cat(z_rows, dim=0).to(device)
    cell = torch.diag(torch.tensor([CELL_SIZE] * 3, device=device))
    comp_frac = torch.tensor([fracs.tolist()], dtype=torch.float32, device=device)

    print(f"Generating {NUM_RUNS} structures: {num_atoms} atoms, {CELL_SIZE} A cell")
    print(f"  Species counts: {dict(zip(SPECIES, counts.tolist()))}")

    r = calc.r_grid.numpy()
    phi_deg = np.rad2deg(calc.phi_grid.numpy())

    if SAVE_STRUCTURES:
        from pathlib import Path
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    gen_g2s = []
    gen_adfs = []

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
        pos = generate(
            cell=cell, num_atoms=num_atoms, z=z,
            velocity_model=lit.ema_model,
            g2_target=target_g2, adf_target=target_adf,
            comp_frac=comp_frac,
            cutoff=CUTOFF, num_steps=NUM_STEPS, method=METHOD,
        )

        atoms = positions_to_atoms(pos, cell, species_gen)

        if SAVE_STRUCTURES:
            outfile = f"{OUTPUT_DIR}/generated_{run:03d}.xyz"
            ase.io.write(outfile, atoms)
            print(f"  Saved: {outfile}")

        # Compute normalized spectra of generated structure
        gen_g2, gen_adf = compute_spectra(atoms, calc, SPECIES)
        gen_g2s.append(gen_g2)
        gen_adfs.append(gen_adf)

        # RMSE on normalized spectra
        g2_err = np.mean((gen_g2 - ref_g2) ** 2) ** 0.5
        adf_err = np.mean((gen_adf - ref_adf) ** 2) ** 0.5
        print(f"  g(r) RMSE: {g2_err:.4f}, ADF RMSE: {adf_err:.4f}")

    # ── Plotting ──────────────────────────────────────────────────────────

    pair_labels = calc.pair_labels
    triplet_labels = calc.triplet_labels
    num_g2_pairs = ref_g2.shape[0] * ref_g2.shape[1]
    num_triplets = ref_adf.shape[0]

    # g(r) comparison
    fig, axs = plt.subplots(1, num_g2_pairs, figsize=(6 * num_g2_pairs, 5))
    if num_g2_pairs == 1:
        axs = [axs]
    idx = 0
    for i in range(ref_g2.shape[0]):
        for j in range(ref_g2.shape[1]):
            ax = axs[idx]
            ax.plot(r, ref_g2[i, j], "k-", lw=2, label="reference")
            for k, gg2 in enumerate(gen_g2s):
                ax.plot(r, gg2[i, j], lw=1.5, alpha=0.7, label=f"gen {k}")
            ax.axhline(1.0, color="gray", ls=":", lw=1)
            ax.set_xlabel("r (A)")
            ax.set_ylabel("g(r)")
            ax.set_title(pair_labels[idx])
            ax.legend(fontsize=8)
            idx += 1
    fig.suptitle("Pair distribution functions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/g2_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR}/g2_comparison.png")
    plt.show()

    # ADF comparison
    ncols = min(num_triplets, 3)
    nrows = (num_triplets + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for t_idx in range(num_triplets):
        ax = axs[t_idx // ncols][t_idx % ncols]
        ax.plot(phi_deg, ref_adf[t_idx], "k-", lw=2, label="reference")
        for k, gadf in enumerate(gen_adfs):
            ax.plot(phi_deg, gadf[t_idx], lw=1.5, alpha=0.7, label=f"gen {k}")
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("P(phi)")
        ax.set_title(triplet_labels[t_idx])
        ax.legend(fontsize=8)
    # Hide unused subplots
    for t_idx in range(num_triplets, nrows * ncols):
        axs[t_idx // ncols][t_idx % ncols].set_visible(False)
    fig.suptitle("Angular distribution functions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/adf_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR}/adf_comparison.png")
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
