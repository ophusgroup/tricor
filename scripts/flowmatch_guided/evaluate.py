"""Evaluate a trained unconditional flow matching model.

Generates structures both with and without guidance, computes
normalized g2/ADF, and compares against a reference.

Edit the CONFIG section below, then run:
    python evaluate.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
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

CHECKPOINT = "./lightning_logs/flowmatch-uncond-si3n4/version_0/checkpoints/last.ckpt"
REFERENCE_STRUCTURE = "/pscratch/sd/e/ehrdt/mcstructgen/smallcell/Si3N4(10)_d90_g3_jit2_s0.15_m2.xyz"
SPECIES = [7, 14]
ATOM_FRACTIONS = [3/7, 4/7]

CELL_SIZE = 25.0
NUM_ATOMS = None                         # if None, match reference

# Spectral settings (for both target computation and evaluation)
R_MAX = 10.0
R_STEP = 0.05
PHI_NUM_BINS = 90
SIGMA_R = 0.15
SIGMA_PHI = 0.1

# Generation settings
CUTOFF = 5.0
UNCOND_STEPS = 30                        # steps for unconditional generation
GUIDED_STEPS = 50                        # steps for guided generation
W = 3000.0                              # guidance weight
NUM_RUNS = 3                             # runs per method

DEVICE = "cuda"
SEED = 42
OUTPUT_DIR = "./eval_output_guided/"

# ══════════════════════════════════════════════════════════════════════════════


def normalize_g2(g2_raw, r, num_center, num_neighbor, volume, dr):
    rho = num_neighbor / volume
    shell_vol = 4.0 * math.pi * r ** 2 * dr
    denom = num_center * rho * shell_vol
    return g2_raw / np.maximum(denom, 1e-12)


def normalize_adf(adf_raw, dphi):
    total = adf_raw.sum() * dphi
    if total > 1e-12:
        return adf_raw / total
    return adf_raw


def compute_normalized_spectra(atoms, calc, species_list):
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

    g2_norm = np.zeros_like(g2_raw)
    for i in range(num_species):
        n_i = (atoms.numbers == species_list[i]).sum()
        for j in range(num_species):
            n_j = (atoms.numbers == species_list[j]).sum()
            g2_norm[i, j] = normalize_g2(g2_raw[i, j], r, n_i, n_j, volume, dr)

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

    ref = ase.io.read(REFERENCE_STRUCTURE)
    print(f"Reference: {len(ref)} atoms, cell={ref.cell.lengths()}")
    num_atoms = NUM_ATOMS if NUM_ATOMS is not None else len(ref)
    num_species = len(SPECIES)

    calc = DifferentiablePDFADF(
        r_max=R_MAX, r_step=R_STEP, phi_num_bins=PHI_NUM_BINS,
        sigma_r=SIGMA_R, sigma_phi=SIGMA_PHI, species=SPECIES,
    ).double()

    # Reference spectra (normalized)
    print("Computing reference spectra...")
    ref_g2, ref_adf = compute_normalized_spectra(ref, calc, SPECIES)

    # Raw targets for guidance
    ref_pos = torch.tensor(ref.positions, dtype=torch.float64, device=device)
    ref_sp = torch.tensor(ref.numbers, dtype=torch.long, device=device)
    ref_cell = torch.tensor(ref.cell.array, dtype=torch.float64, device=device)
    with torch.no_grad():
        target_g2, target_adf = calc.compute(ref_pos, ref_sp, ref_cell)

    loss_fn = DifferentiableSpectralLoss(calc)

    # Load model
    print(f"Loading: {CHECKPOINT}")
    lit = LitUncondFlowMatch.load_from_checkpoint(CHECKPOINT, map_location=device)
    lit.ema_model.to(device)
    lit.ema_model.eval()

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

    from pathlib import Path
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    r = calc.r_grid.numpy()
    phi_deg = np.rad2deg(calc.phi_grid.numpy())

    # ── Generate: unconditional ──────────────────────────────────────────

    print(f"\n{'='*60}")
    print("UNCONDITIONAL GENERATION (no guidance)")
    print(f"{'='*60}")

    uncond_g2s, uncond_adfs = [], []
    for run in range(NUM_RUNS):
        print(f"  Run {run+1}/{NUM_RUNS}")
        pos = generate_unconditional(
            cell, num_atoms, z, lit.ema_model,
            cutoff=CUTOFF, num_steps=UNCOND_STEPS,
        )
        atoms = positions_to_atoms(pos, cell, species)
        ase.io.write(f"{OUTPUT_DIR}/uncond_{run:03d}.xyz", atoms)

        g2_n, adf_n = compute_normalized_spectra(atoms, calc, SPECIES)
        uncond_g2s.append(g2_n)
        uncond_adfs.append(adf_n)

        g2_err = np.mean((g2_n - ref_g2) ** 2) ** 0.5
        adf_err = np.mean((adf_n - ref_adf) ** 2) ** 0.5
        print(f"    g(r) RMSE: {g2_err:.4f}, ADF RMSE: {adf_err:.4f}")

    # ── Generate: with guidance ──────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"GUIDED GENERATION (w={W})")
    print(f"{'='*60}")

    guided_g2s, guided_adfs = [], []
    for run in range(NUM_RUNS):
        print(f"  Run {run+1}/{NUM_RUNS}")
        pos = generate_guided(
            cell, num_atoms, z, lit.ema_model,
            loss_fn, target_g2, target_adf, species,
            cutoff=CUTOFF, w=W, num_steps=GUIDED_STEPS,
            verbose=True,
        )
        atoms = positions_to_atoms(pos, cell, species)
        ase.io.write(f"{OUTPUT_DIR}/guided_{run:03d}.xyz", atoms)

        g2_n, adf_n = compute_normalized_spectra(atoms, calc, SPECIES)
        guided_g2s.append(g2_n)
        guided_adfs.append(adf_n)

        g2_err = np.mean((g2_n - ref_g2) ** 2) ** 0.5
        adf_err = np.mean((adf_n - ref_adf) ** 2) ** 0.5
        print(f"    g(r) RMSE: {g2_err:.4f}, ADF RMSE: {adf_err:.4f}")

    # ── Plotting ─────────────────────────────────────────────────────────

    pair_labels = calc.pair_labels
    triplet_labels = calc.triplet_labels
    num_pairs = ref_g2.shape[0] * ref_g2.shape[1]
    num_triplets = ref_adf.shape[0]

    # g(r) comparison
    fig, axs = plt.subplots(1, num_pairs, figsize=(6 * num_pairs, 5))
    if num_pairs == 1:
        axs = [axs]
    idx = 0
    for i in range(ref_g2.shape[0]):
        for j in range(ref_g2.shape[1]):
            ax = axs[idx]
            ax.plot(r, ref_g2[i, j], "k-", lw=2.5, label="reference")
            for k, gg2 in enumerate(uncond_g2s):
                ax.plot(r, gg2[i, j], lw=1, alpha=0.5, color="tab:blue",
                        label="uncond" if k == 0 else None)
            for k, gg2 in enumerate(guided_g2s):
                ax.plot(r, gg2[i, j], lw=1.5, alpha=0.7, color="tab:red",
                        label="guided" if k == 0 else None)
            ax.axhline(1.0, color="gray", ls=":", lw=1)
            ax.set_xlabel("r (A)")
            ax.set_ylabel("g(r)")
            ax.set_title(pair_labels[idx])
            ax.legend(fontsize=9)
            idx += 1
    fig.suptitle("Pair distribution functions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/g2_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR}/g2_comparison.png")

    # ADF comparison
    ncols = min(num_triplets, 3)
    nrows = (num_triplets + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for t_idx in range(num_triplets):
        ax = axs[t_idx // ncols][t_idx % ncols]
        ax.plot(phi_deg, ref_adf[t_idx], "k-", lw=2.5, label="reference")
        for k, gadf in enumerate(uncond_adfs):
            ax.plot(phi_deg, gadf[t_idx], lw=1, alpha=0.5, color="tab:blue",
                    label="uncond" if k == 0 else None)
        for k, gadf in enumerate(guided_adfs):
            ax.plot(phi_deg, gadf[t_idx], lw=1.5, alpha=0.7, color="tab:red",
                    label="guided" if k == 0 else None)
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("P(phi)")
        ax.set_title(triplet_labels[t_idx])
        ax.legend(fontsize=8)
    for t_idx in range(num_triplets, nrows * ncols):
        axs[t_idx // ncols][t_idx % ncols].set_visible(False)
    fig.suptitle("Angular distribution functions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/adf_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR}/adf_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
