"""Probe how softening the MACE+wall potential changes:
  (a) per-step displacement magnitudes available for training
  (b) fmax descent behavior (does the wall still tame MACE?)
  (c) structural quality at the final frame — critically, the 50° Si-O-Si
      artifact that motivated the wall in the first place

If we soften the wall too much, MACE drifts into the spurious low-energy
basin and the structures pick up the very artifacts the wall existed to
prevent.  The PDF / ADF overlay is the litmus test.

For each (k, exponent) config, runs 80 FIRE steps with maxstep=0.5
(the more-aggressive setting that the previous probe identified as the
best for getting larger displacements).  At step 80, computes g(r) for
all three Si/O pair types and the ADF for Si-centered and O-centered
triplets.  Overlays all configs on one plot for easy comparison.

The key artifact diagnostics (printed + in CSV):
  - Si-Si first-peak position   (physical ~3.06 Å for α-quartz; the
                                 artifact pushes Si pairs to ~1.5-2 Å)
  - Si-O first-peak position    (physical ~1.61 Å)
  - ADF Si-centered low-angle intensity (40-70°) — the 50° hump

Outputs:
  scratch/probe_fire_wall_summary.csv
  scratch/probe_fire_wall_traces.csv      (per-step E/fmax/dr)
  scratch/probe_fire_wall_pdf_adf.png     (overlay)

Run (pick a free GPU):
    CUDA_VISIBLE_DEVICES=0 \\
      /home/ehrdt/miniforge3/envs/mace/bin/python scratch/probe_fire_wall.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read
from ase.optimize import FIRE

import tricor as tc
from mace.calculators import mace_mp
from tricor.differentiable_pdf_fast import DifferentiablePDFADF_Fast

sys.path.insert(0, "/home/ehrdt/tricor/mace")
from wall_calculator import MinDistanceWallCalculator, per_pair_min_from_atoms  # noqa: E402


# === CONFIG ============================================================
REFERENCE_CIF = Path("/wigeon/users/ehrdt/prod/cifs_mp_cnos/mp-7000_SiO2.cif")
CELL_DIM = 50.0
RELATIVE_DENSITY = 0.92
REGIME = "amorphous"
RNG_SEED = 42
N_STEPS = 80
OPT_MAXSTEP = 0.5
WALL_MARGIN = 0.0

# PDF / ADF settings (lifted from sweep_regimes_sio2.py).
PDF_R_MAX = 10.0
PDF_R_STEP = 0.05
PDF_PHI_BINS = 90
ADF_R_MAX = 2.2
PLOT_R_MAX = 6.0
SPECIES = [8, 14]  # O first, Si second

OUT_DIR = Path("/home/ehrdt/tricor/scratch")
SUMMARY_CSV = OUT_DIR / "probe_fire_wall_summary.csv"
TRACE_CSV   = OUT_DIR / "probe_fire_wall_traces.csv"
PDF_PNG     = OUT_DIR / "probe_fire_wall_pdf_adf.png"

# (label, wall_k, wall_exponent).  Production baseline first for reference;
# subsequent configs progressively soften the wall.  k=0 means "no wall" —
# this should show MACE's untamed artifact behavior.
WALL_CONFIGS = [
    ("baseline_k1000_n4",  1000.0, 4),   # current production
    ("k300_n4",             300.0, 4),
    ("k100_n4",             100.0, 4),
    ("k1000_n2",           1000.0, 2),   # quadratic — softer near threshold
    ("k300_n2",             300.0, 2),
    ("k100_n2",             100.0, 2),
    ("no_wall",               0.0, 4),   # MACE alone — sanity baseline
]
# =======================================================================


def _min_image(delta, cell):
    inv = np.linalg.inv(cell)
    df = delta @ inv.T
    df -= np.round(df)
    return df @ cell


def build_initial_state():
    ref = read(REFERENCE_CIF)
    shell = tc.CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)
    sc = tc.Supercell.from_atoms(
        ref, cell_dim_angstroms=CELL_DIM,
        r_max=10, r_step=0.1, phi_num_bins=90,
        relative_density=RELATIVE_DENSITY, rng_seed=RNG_SEED,
        label=f"probe_{REGIME}",
    )
    preset = dict(tc.Supercell.PRESETS[REGIME])
    preset["displacement_sigma"] = 0.0
    sc.generate(shell, **{**preset, "num_steps": 0},
                refine_orientations=False, show_progress=False)
    sc.bond_relax(shell, n_iter=80, max_step=0.1)
    return sc.atoms


def _pdf_adf_module(species, device, dtype=torch.float64):
    mod = DifferentiablePDFADF_Fast(
        r_max=PDF_R_MAX, r_step=PDF_R_STEP,
        phi_num_bins=PDF_PHI_BINS, species=species, adf_r_max=ADF_R_MAX,
    ).to(device=device, dtype=dtype)
    return mod


def _compute_pdf_adf(mod, atoms, device, dtype=torch.float64):
    pos = torch.as_tensor(atoms.positions, dtype=dtype, device=device)
    z = torch.as_tensor(atoms.numbers, dtype=torch.int64, device=device)
    cell = torch.as_tensor(atoms.cell.array, dtype=dtype, device=device)
    with torch.no_grad():
        g2, adf = mod.compute(pos, z, cell)
    return g2.detach().cpu().numpy(), adf.detach().cpu().numpy()


def _grids():
    num_r = int(round(PDF_R_MAX / PDF_R_STEP))
    r = np.arange(num_r) * PDF_R_STEP + 0.5 * PDF_R_STEP
    phi_edges = np.linspace(0.0, np.pi, PDF_PHI_BINS + 1)
    phi = phi_edges[:-1] + 0.5 * (phi_edges[1] - phi_edges[0])
    return r, np.rad2deg(phi)


def _gofr(g_pair, ni, nj, V, r_grid):
    denom = ni * (ni - 1) if ni == nj else ni * nj
    if denom <= 0:
        return g_pair * 0.0
    shell = 4.0 * np.pi * np.maximum(r_grid, 1e-6) ** 2 * PDF_R_STEP
    return g_pair * V / (denom * shell)


def _area_norm(y, x):
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = float(_trapz(y, x))
    return y / area if area > 0 else y


def _first_peak(x, y, x_min):
    from scipy.signal import find_peaks
    mask = x >= x_min
    xu, yu = x[mask], y[mask]
    if yu.max() <= 0:
        return None, None
    idx, _ = find_peaks(yu, prominence=0.05 * yu.max())
    if len(idx) == 0:
        return None, None
    return float(xu[idx[0]]), float(yu[idx[0]])


def _adf_low_angle_mass(adf_curve, phi_grid, lo_deg=40.0, hi_deg=70.0):
    """Integrate the area-normalized ADF over [lo, hi] degrees.
    Larger = more low-angle (artifact) intensity."""
    yn = _area_norm(adf_curve, phi_grid)
    mask = (phi_grid >= lo_deg) & (phi_grid <= hi_deg)
    if not mask.any():
        return 0.0
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(yn[mask], phi_grid[mask]))


def run_one(label, wall_k, wall_exp, base_calc, atoms_template,
             pdf_mod, device, trace_writer):
    atoms = Atoms(
        numbers=atoms_template.numbers,
        positions=atoms_template.positions.copy(),
        cell=atoms_template.cell.array.copy(),
        pbc=atoms_template.pbc,
    )

    if wall_k > 0:
        r_min = per_pair_min_from_atoms(atoms, margin=WALL_MARGIN)
        atoms.calc = MinDistanceWallCalculator(
            base_calc=base_calc, r_min_per_pair=r_min,
            k=wall_k, exponent=wall_exp,
        )
    else:
        atoms.calc = base_calc                 # untamed MACE

    e0 = float(atoms.get_potential_energy())
    f0 = float(np.abs(atoms.get_forces()).max())
    prev_pos = atoms.positions.copy()

    per_step_dr_mean, per_step_dr_max = [], []
    per_step_e, per_step_f = [e0], [f0]

    def cb():
        nonlocal prev_pos
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        dxs = _min_image(atoms.positions - prev_pos, cell)
        dr = np.linalg.norm(dxs, axis=-1)
        per_step_dr_mean.append(float(dr.mean()))
        per_step_dr_max.append(float(dr.max()))
        per_step_e.append(float(atoms.get_potential_energy()))
        per_step_f.append(float(np.abs(atoms.get_forces()).max()))
        prev_pos = atoms.positions.copy()

    opt = FIRE(atoms, maxstep=OPT_MAXSTEP, logfile=None)
    opt.attach(cb, interval=1)
    t0 = time.perf_counter()
    opt.run(fmax=0.01, steps=N_STEPS)
    elapsed = time.perf_counter() - t0

    for step in range(len(per_step_dr_mean)):
        trace_writer.writerow({
            "config": label,
            "step": step + 1,
            "dr_mean": per_step_dr_mean[step],
            "dr_max":  per_step_dr_max[step],
            "energy": per_step_e[step + 1],
            "fmax":   per_step_f[step + 1],
        })

    # PDF + ADF at final frame
    g2, adf = _compute_pdf_adf(pdf_mod, atoms, device)
    return {
        "label": label,
        "wall_k": wall_k,
        "wall_exp": wall_exp,
        "n_atoms": int(len(atoms)),
        "elapsed_s": elapsed,
        "e_start": e0,
        "e_end":   per_step_e[-1],
        "delta_e": per_step_e[-1] - e0,
        "fmax_start": f0,
        "fmax_end":   per_step_f[-1],
        "dr_mean_avg":   float(np.mean(per_step_dr_mean)),
        "dr_mean_early": float(np.mean(per_step_dr_mean[:10])),
        "dr_mean_late":  float(np.mean(per_step_dr_mean[-10:])),
        "dr_max_overall": float(np.max(per_step_dr_max)),
        "g2": g2,
        "adf": adf,
        "z": np.asarray(atoms.numbers),
        "V": float(abs(np.linalg.det(np.asarray(atoms.cell.array)))),
    }


def _make_overlay(results, out_png):
    r_grid, phi_grid = _grids()
    O_i, Si_i = 0, 1
    pair_panels = [
        ("Si-Si", Si_i, Si_i),
        ("Si-O",  Si_i, O_i),
        ("O-O",   O_i,  O_i),
    ]
    adf_panels = [
        ("ADF Si-centered", [3, 4, 5]),
        ("ADF O-centered",  [0, 1, 2]),
    ]
    cmap = plt.get_cmap("viridis")
    colors = {r["label"]: cmap(i / max(1, len(results) - 1))
              for i, r in enumerate(results)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (lbl, i, j) in enumerate(pair_panels):
        ax = axes[0, col]
        for r in results:
            ni = int((r["z"] == SPECIES[i]).sum())
            nj = int((r["z"] == SPECIES[j]).sum())
            y = _gofr(r["g2"][i, j], ni, nj, r["V"], r_grid)
            ax.plot(r_grid, y, color=colors[r["label"]], lw=1.5, label=r["label"])
        ax.set_xlim(0.0, PLOT_R_MAX)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"g(r): {lbl}")
        ax.axhline(1.0, color="0.7", lw=0.7, ls=":")
        if col == 0:
            ax.legend(framealpha=0.9, fontsize=8)

    for col, (title, idxs) in enumerate(adf_panels):
        ax = axes[1, col]
        for r in results:
            adf_sum = r["adf"][idxs].sum(axis=0)
            ax.plot(phi_grid, _area_norm(adf_sum, phi_grid),
                    color=colors[r["label"]], lw=1.5, label=r["label"])
        ax.axvspan(40, 70, color="red", alpha=0.10,
                   label="artifact band (40-70°)" if col == 0 else None)
        ax.set_xlabel("bond angle φ (deg)")
        ax.set_ylabel("ADF(φ) [normalized]")
        ax.set_title(title)
        if col == 0:
            ax.legend(framealpha=0.9, fontsize=8)
    axes[1, 2].axis("off")

    fig.suptitle("Wall sweep on amorphous SiO2 (MACE+wall+FIRE, maxstep=0.5, N=80) — "
                 "g(r) and ADF at step 80", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _peak_summary(results):
    """First-peak position/height per panel + ADF low-angle integral."""
    r_grid, phi_grid = _grids()
    O_i, Si_i = 0, 1
    out_rows = []
    for r in results:
        ni_si = int((r["z"] == SPECIES[Si_i]).sum())
        ni_o  = int((r["z"] == SPECIES[O_i]).sum())
        # Per-pair g(r) peaks (look past r=1 Å so noise doesn't trick the finder).
        sisi = _gofr(r["g2"][Si_i, Si_i], ni_si, ni_si, r["V"], r_grid)
        sio  = _gofr(r["g2"][Si_i, O_i],  ni_si, ni_o,  r["V"], r_grid)
        oo   = _gofr(r["g2"][O_i,  O_i],  ni_o,  ni_o,  r["V"], r_grid)
        sisi_x, sisi_y = _first_peak(r_grid, sisi, x_min=1.0)
        sio_x,  sio_y  = _first_peak(r_grid, sio,  x_min=0.8)
        oo_x,   oo_y   = _first_peak(r_grid, oo,   x_min=1.5)
        # ADF low-angle mass (40-70°), Si-centered and O-centered.
        adf_si = r["adf"][[3, 4, 5]].sum(axis=0)
        adf_o  = r["adf"][[0, 1, 2]].sum(axis=0)
        si_low = _adf_low_angle_mass(adf_si, phi_grid)
        o_low  = _adf_low_angle_mass(adf_o,  phi_grid)
        out_rows.append({
            "config":          r["label"],
            "wall_k":          r["wall_k"],
            "wall_exp":        r["wall_exp"],
            "dr_mean_avg":     round(r["dr_mean_avg"], 5),
            "dr_max_overall":  round(r["dr_max_overall"], 4),
            "fmax_start":      round(r["fmax_start"], 3),
            "fmax_end":        round(r["fmax_end"], 3),
            "delta_e":         round(r["delta_e"], 1),
            "sisi_peak_r":     round(sisi_x, 3) if sisi_x else None,
            "sisi_peak_y":     round(sisi_y, 3) if sisi_y else None,
            "sio_peak_r":      round(sio_x, 3) if sio_x else None,
            "sio_peak_y":      round(sio_y, 3) if sio_y else None,
            "oo_peak_r":       round(oo_x, 3) if oo_x else None,
            "oo_peak_y":       round(oo_y, 3) if oo_y else None,
            "adf_si_low":      round(si_low, 4),  # artifact metric (40-70°)
            "adf_o_low":       round(o_low, 4),
        })
    return out_rows


def main():
    print(f"Wall-parameter probe — amorphous SiO2, MACE+wall+FIRE, "
          f"maxstep={OPT_MAXSTEP}, steps={N_STEPS}")
    print(f"Configs: {[c[0] for c in WALL_CONFIGS]}")

    print("\nBuilding initial structure (one-shot, reused across configs)...")
    atoms_template = build_initial_state()
    print(f"  {len(atoms_template)} atoms")

    print("\nInitializing MACE-MPA medium...")
    base_calc = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float32")
    print("MACE ready.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdf_mod = _pdf_adf_module(SPECIES, device=device)
    print(f"PDF/ADF compute device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    with open(TRACE_CSV, "w", newline="") as tf:
        trace_writer = csv.DictWriter(
            tf,
            fieldnames=["config", "step", "dr_mean", "dr_max", "energy", "fmax"],
            lineterminator="\n",
        )
        trace_writer.writeheader()
        for label, k, n_exp in WALL_CONFIGS:
            print(f"\n=== {label}  (k={k}, exp={n_exp}) ===")
            try:
                r = run_one(label, k, n_exp, base_calc, atoms_template,
                             pdf_mod, device, trace_writer)
                results.append(r)
                print(
                    f"  steps=80  |Δr| avg={r['dr_mean_avg']:.5f}  "
                    f"max={r['dr_max_overall']:.4f}  "
                    f"E {r['e_start']:.1f}→{r['e_end']:.1f}  "
                    f"ΔE={r['delta_e']:+.1f}  "
                    f"fmax {r['fmax_start']:.2f}→{r['fmax_end']:.2f}  "
                    f"{r['elapsed_s']:.1f}s"
                )
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        return

    print(f"\nGenerating overlay PNG: {PDF_PNG}")
    _make_overlay(results, PDF_PNG)

    print("\nFirst-peak summary (the SiO2-physical Si-O is 1.61 Å; Si-Si ~3.06 Å,")
    print("O-O ~2.64 Å for α-quartz.  The MACE artifact pulls Si-Si below 2 Å")
    print("and creates a 50° ADF Si-centered hump → 'adf_si_low' grows.):")
    rows = _peak_summary(results)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {SUMMARY_CSV}")

    # Compact diagnostic table
    print("\n  config                |Δr|avg  ΔE       fmax→  "
          "Si-Si@r  Si-O@r  ADF-Si low(40-70°)")
    for r in rows:
        sisi = f"{r['sisi_peak_r']:.2f}" if r["sisi_peak_r"] else "  —"
        sio  = f"{r['sio_peak_r']:.2f}"  if r["sio_peak_r"]  else "  —"
        print(f"  {r['config']:<22s} {r['dr_mean_avg']:.5f}  "
              f"{r['delta_e']:+7.0f}  {r['fmax_end']:5.2f}  "
              f"{sisi:>5s}    {sio:>5s}   {r['adf_si_low']:.4f}")


if __name__ == "__main__":
    main()
