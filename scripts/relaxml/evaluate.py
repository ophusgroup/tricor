"""Iterative inference + comparison against tricor ground truth.

Loads a trained LitRelaxML checkpoint, applies it iteratively to its
own output on a target .npz sample until convergence, then compares
the predicted final structure to the tricor-generated ``best_positions``.

Edit the CONFIG block below, then run:
    python evaluate.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (mallard caps + paths must be set before torch import)
# ─────────────────────────────────────────────────────────────────────────────

# --- mallard resource caps ---
GPU_ID = 2
NUM_THREADS = 2

# --- what to evaluate ---
# Path to the Lightning checkpoint to load.  Lightning writes to
# ./lightning_logs/<RUN_NAME>/version_<N>/checkpoints/.
CHECKPOINT = "./lightning_logs/relaxml-si/version_5/checkpoints/last.ckpt"

# Architecture variant used to train the checkpoint above.
#   "current" — multi-species model (nn.Embedding node encoder + species
#               pair-embedding edge features).
#   "v1"      — original single-species model (one-hot MLP node encoder,
#               no species pair edge features).  Use this for checkpoints
#               from before the multi-species refactor.
MODEL_VERSION = "v1"

# Path to either a single .npz file or a directory of .npz files.
TARGET =  "./data/si_test_cells/cell080/si_liquid_cell080_idx00001_seed000300001.npz" 

# When True and TARGET is a directory, pick one .npz at random per regime
# instead of evaluating every file.  Useful for quick sanity checks
# (6 evaluations total).  When False, evaluates every .npz in TARGET.
SAMPLE_PER_REGIME = True
SAMPLE_SEED = 0                  # reproducibility for the per-regime random pick

REGIMES = (
    "liquid", "amorphous", "SRO", "MRO", "LRO", "nanocrystalline",
)

# --- inference controls ---
MAX_ITER = 50
CONVERGENCE_TOL_ANG = 0.01       # Å; stop when max per-atom displacement < tol
USE_EMA_WEIGHTS = True           # use the EMA snapshot of the model
SPECIES = [14]
CUTOFF = 5.0

# --- optional tricor "finetune" after ML inference ---
# Run this many additional shell_relax steps with the original weight
# parameters after the iterative ML inference converges.  0 disables.
# 5–20 is the useful range — enough to clean up small residual errors
# at the disorder extremes without losing the wall-time win.  Uses the
# weight params stored in the .npz (so behavior matches what tricor
# would have done from the start, just on a much-better starting state).
TRICOR_FINETUNE_STEPS = 200


# --- PDF / ADF comparison ---
# PDF_R_MAX is computed slightly past the 8 Å plot crop so that the
# Gaussian-tail clipping artifact at the neighbor-list cutoff edge stays
# off-screen.  Bump together if PLOT_R_MAX changes.
PDF_R_MAX = 10.0
PDF_R_STEP = 0.05
PDF_PHI_BINS = 90
PLOT_R_MAX = 8.0                 # x-axis limit for the g(r) plot panel

# Set False to skip the PDF/ADF metric computation entirely (also disables
# plotting, since plots reuse those arrays).  Useful for pure timing runs.
EVAL_METRICS = True

# --- visualization ---
# Save 2-panel PNGs (g2 + ADF) overlaying predicted vs tricor target for
# each evaluated structure.  Initial-state curve is also drawn faintly for
# context.  Output goes to PLOT_DIR (None = auto: <TARGET>/evaluation_plots/).
SAVE_PLOTS = True
PLOT_DIR = None
INCLUDE_INITIAL_IN_PLOTS = True

# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
_n = str(NUM_THREADS)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = _n

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
torch.set_num_threads(NUM_THREADS)
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, Batch

from tricor.flowmatch.flow_utils import (
    periodic_radius_graph_cell_list,
    periodic_radius_graph_chunked,
)

if MODEL_VERSION == "current":
    from tricor.relaxml import LitRelaxML
elif MODEL_VERSION == "v1":
    from tricor.relaxml.model_v1 import LitRelaxML
else:
    raise ValueError(f"Unknown MODEL_VERSION: {MODEL_VERSION!r}")
from tricor.relaxml.data import (
    _min_image_displacement,
    _weight_vector_from_row,
    WEIGHT_FEATURE_KEYS,
)


def _periodic_graph(pos, cutoff, cell):
    try:
        return periodic_radius_graph_cell_list(pos, cutoff, cell)
    except ValueError:
        return periodic_radius_graph_chunked(pos, cutoff, cell=cell)


def _load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    lit = LitRelaxML.load_from_checkpoint(str(ckpt_path), map_location=device)
    lit.eval()
    lit.to(device)
    # Pull the EMA weights into the main model if requested — these are
    # generally what you want for inference.
    if USE_EMA_WEIGHTS and hasattr(lit, "ema_model"):
        lit.ema_model.eval()
        return lit.ema_model.module.to(device)
    return lit.model.to(device)


def _build_data(
    positions: torch.Tensor,
    cell: torch.Tensor,
    z_onehot: torch.Tensor,
    weight_vector: torch.Tensor,
    cutoff: float,
) -> Batch:
    edge_index, edge_vec = _periodic_graph(positions, cutoff, cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])
    data = Data(
        z=z_onehot,
        pos=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
        w=weight_vector.unsqueeze(0),
    )
    return Batch.from_data_list([data])


def _weight_vector_from_npz(npz) -> np.ndarray:
    fake_row = {k: str(float(npz[k])) for k in (
        "bond_weight", "angle_weight", "repulsion_weight",
        "hard_core_scale", "nonbond_push_scale", "displacement_sigma",
        "grain_size", "crystalline_fraction",
    )}
    fake_row["num_grains"] = str(int(npz["num_grains"]))
    return _weight_vector_from_row(fake_row)


def _wrap_positions(pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Fold positions back into the fundamental cell [0, L)^3."""
    inv_cell = torch.linalg.inv(cell)
    frac = pos @ inv_cell.T
    frac = frac - torch.floor(frac)
    return frac @ cell


@torch.no_grad()
def run_iterative_inference(
    model: torch.nn.Module,
    initial_positions: np.ndarray,
    cell: np.ndarray,
    species: np.ndarray,
    weight_vector: np.ndarray,
    device: torch.device,
    cutoff: float = CUTOFF,
    max_iter: int = MAX_ITER,
    tol: float = CONVERGENCE_TOL_ANG,
) -> tuple[np.ndarray, int]:
    """Apply the model iteratively until convergence; return (final_pos, n_iter)."""
    pos = torch.tensor(initial_positions, dtype=torch.float32, device=device)
    cell_t = torch.tensor(cell, dtype=torch.float32, device=device)
    w_t = torch.tensor(weight_vector, dtype=torch.float32, device=device)

    unique_numbers = np.array(sorted(SPECIES))
    atom_encoder = OneHotEncoder(sparse_output=False)
    atom_encoder.fit(unique_numbers.reshape(-1, 1))
    z_onehot = torch.tensor(
        atom_encoder.transform(species.reshape(-1, 1)),
        dtype=torch.float32, device=device,
    )

    for it in range(max_iter):
        batch = _build_data(pos, cell_t, z_onehot, w_t, cutoff)
        delta = model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
        )
        # Diagnostic: report the magnitude of model-predicted per-atom
        # displacements at every iteration.  Compared against the training
        # target which has |delta|.mean() ≈ 0.1 Å — if these come out
        # orders-of-magnitude larger, the model is mis-applied at inference.
        d_norms = delta.norm(dim=-1)
        print(
            f"  iter {it:2d}: |delta| mean={d_norms.mean().item():.4f}  "
            f"max={d_norms.max().item():.4f}  "
            f"std={d_norms.std().item():.4f}  Å"
        )
        pos_new = _wrap_positions(pos + delta, cell_t)
        max_step = (pos_new - pos).norm(dim=-1).max().item()
        pos = pos_new
        if max_step < tol:
            return pos.cpu().numpy(), it + 1
    return pos.cpu().numpy(), max_iter


def _tricor_finetune(
    positions: np.ndarray,
    species: np.ndarray,
    cell: np.ndarray,
    weights: dict,
    n_steps: int,
) -> np.ndarray:
    """Run ``n_steps`` of tricor's shell_relax starting from ``positions``.

    Used as a post-ML cleanup pass: the ML model gets close, then a small
    number of native tricor steps polishes residual errors using the
    original weight parameters from the source .npz.  Returns the new
    positions (best-loss snapshot from the relaxation, since that's what
    shell_relax restores into ``self.atoms``).

    No-op if ``n_steps <= 0``.
    """
    if n_steps <= 0:
        return positions

    from ase.atoms import Atoms
    from ase.build import bulk
    from tricor.shells import CoordinationShellTarget
    from tricor.supercell import Supercell

    ref = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
    shell_target = CoordinationShellTarget.from_atoms(ref)

    cell_edge = tuple(float(x) for x in np.diag(cell))
    sc = Supercell.from_atoms(
        ref, cell_dim_angstroms=cell_edge, rng_seed=0, relative_density=0.96,
    )
    # Overwrite the random Supercell.from_atoms placement with our
    # ML-predicted positions; refresh cached cell matrices.
    sc.atoms = Atoms(
        numbers=species, positions=positions, cell=cell, pbc=ref.pbc,
    )
    sc._cell_matrix = np.asarray(sc.atoms.cell.array, dtype=np.float64)
    sc._cell_inverse = np.linalg.inv(sc._cell_matrix)

    sc.shell_relax(
        shell_target,
        num_steps=int(n_steps),
        show_progress=False,
        **weights,
    )
    return np.asarray(sc.atoms.positions, dtype=np.float32).copy()


def _pdf_adf(positions: np.ndarray, species: np.ndarray, cell: np.ndarray):
    """Compute g2 + ADF on a single structure.  Lazy import to keep main fast."""
    from tricor.differentiable_pdf_fast import DifferentiablePDFADF_Fast
    mod = DifferentiablePDFADF_Fast(
        r_max=PDF_R_MAX, r_step=PDF_R_STEP,
        phi_num_bins=PDF_PHI_BINS, species=SPECIES,
    ).to(torch.float64)
    with torch.no_grad():
        g2, adf = mod.compute(
            torch.tensor(positions, dtype=torch.float64),
            torch.tensor(species, dtype=torch.int64),
            torch.tensor(cell, dtype=torch.float64),
        )
    return g2.numpy(), adf.numpy()


def _pdf_adf_grids() -> tuple[np.ndarray, np.ndarray]:
    """r-axis (Å) and ADF phi-axis (deg) values for the configured PDF settings."""
    num_r = int(round(PDF_R_MAX / PDF_R_STEP))
    r_grid = np.arange(num_r, dtype=np.float64) * PDF_R_STEP + 0.5 * PDF_R_STEP
    phi_edges = np.linspace(0.0, np.pi, PDF_PHI_BINS + 1)
    phi_centers = phi_edges[:-1] + 0.5 * (phi_edges[1] - phi_edges[0])
    return r_grid, np.rad2deg(phi_centers)


def _save_comparison_plot(
    out_path: Path,
    g2_tgt: np.ndarray,
    g2_pred: np.ndarray,
    g2_init: np.ndarray | None,
    adf_tgt: np.ndarray,
    adf_pred: np.ndarray,
    adf_init: np.ndarray | None,
    title: str,
    species: np.ndarray,
    cell: np.ndarray,
) -> None:
    """Two-panel plot: g(r) and ADF(φ), tricor target vs predicted (+ optional initial).

    g(r) uses the standard pair-correlation normalization:
        g(r) = count(r) * V / (N * (N-1) * 4π r² * dr)
    so a uniform random arrangement gives g(r) → 1 at large r.

    ADF stays area-normalized (probability density over φ).
    """
    # Lazy import + Agg backend so headless mallard sessions never need a display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_grid, phi_grid = _pdf_adf_grids()

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

    # Cell volume and per-species counts for the g(r) density normalization.
    # Si-only path: g2[0, 0] is the Si–Si pair; n_per is just N_Si.
    cell_arr = np.asarray(cell, dtype=np.float64)
    V = float(abs(np.linalg.det(cell_arr)))
    species_arr = np.asarray(species, dtype=np.int64)
    sorted_species = sorted(SPECIES)
    n_per = np.array([(species_arr == Z).sum() for Z in sorted_species],
                     dtype=np.float64)
    # 4π r² * dr.  Floor on r to keep the r→0 bin from blowing up; that
    # bin is empty in practice (hard-core repulsion) so the value there
    # is irrelevant.
    shell = 4.0 * np.pi * np.maximum(r_grid, 1e-6) ** 2 * PDF_R_STEP

    def _gofr(g_pair: np.ndarray, i: int, j: int) -> np.ndarray:
        ni, nj = n_per[i], n_per[j]
        denom = ni * (ni - 1) if i == j else ni * nj
        if denom <= 0:
            return g_pair * 0.0
        return g_pair * V / (denom * shell)

    def _area_norm(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        area = float(_trapz(y, x))
        return y / area if area > 0 else y

    g2_t = _gofr(g2_tgt[0, 0], 0, 0)
    g2_p = _gofr(g2_pred[0, 0], 0, 0)
    adf_t = _area_norm(adf_tgt[0], phi_grid)
    adf_p = _area_norm(adf_pred[0], phi_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if g2_init is not None:
        ax.plot(r_grid, _gofr(g2_init[0, 0], 0, 0), color="0.65", lw=1.0,
                ls="--", label="initial (pre-relax)")
    ax.plot(r_grid, g2_t, color="C3", lw=2.0, label="tricor (target)")
    ax.plot(r_grid, g2_p, color="C0", lw=1.6, label="model (predicted)")
    ax.set_xlim(0.0, PLOT_R_MAX)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("Pair distribution")
    ax.axhline(1.0, color="0.7", lw=0.7, ls=":")
    ax.legend(framealpha=0.9, fontsize=9)

    ax = axes[1]
    if adf_init is not None:
        ax.plot(phi_grid, _area_norm(adf_init[0], phi_grid), color="0.65", lw=1.0,
                ls="--", label="initial (pre-relax)")
    ax.plot(phi_grid, adf_t, color="C3", lw=2.0, label="tricor (target)")
    ax.plot(phi_grid, adf_p, color="C0", lw=1.6, label="model (predicted)")
    ax.set_xlabel("bond angle φ (deg)")
    ax.set_ylabel("ADF(φ)  [normalized]")
    ax.set_title("Angle distribution")
    ax.legend(framealpha=0.9, fontsize=9)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def evaluate_one(
    model: torch.nn.Module, npz_path: Path, device: torch.device,
    plot_dir: Path | None = None,
) -> dict:
    with np.load(npz_path) as npz:
        # Use the first trajectory snapshot, NOT npz["initial_positions"].
        # The latter is the random Supercell.from_atoms placement, captured
        # before sc.generate() replaces atoms with grain-Voronoi-built ones
        # for any non-liquid config.  positions[0] is the actual state at
        # step 0 of shell_relax, which is what we trained the model to step
        # forward from.
        initial = np.asarray(npz["positions"][0], dtype=np.float32)
        best = np.asarray(npz["best_positions"], dtype=np.float32)
        cell = np.asarray(npz["cell"], dtype=np.float32)
        species = np.asarray(npz["species_numbers"], dtype=np.int64)
        weight_vector = _weight_vector_from_npz(npz)
        regime = str(npz["regime"].item())
        best_loss = float(npz["best_loss"])
        # Un-normalized weight kwargs for the optional tricor finetune.
        # These are the same values that drove the original tricor
        # relaxation that produced best_positions.  NOTE: displacement_sigma
        # is intentionally NOT passed — it controls thermal jitter during
        # grain CONSTRUCTION (in _build_grain_atoms) and is not a parameter
        # of shell_relax itself.
        finetune_kwargs = {
            "bond_weight":         float(npz["bond_weight"]),
            "angle_weight":        float(npz["angle_weight"]),
            "repulsion_weight":    float(npz["repulsion_weight"]),
            "hard_core_scale":     float(npz["hard_core_scale"]),
            "nonbond_push_scale":  float(npz["nonbond_push_scale"]),
        }

    t0 = time.perf_counter()
    predicted, n_iter = run_iterative_inference(
        model, initial, cell, species, weight_vector, device,
    )
    t_ml = time.perf_counter() - t0

    # Optional tricor finetune.  Runs N steps of native shell_relax with
    # the source-file weights from the ML-predicted starting state.
    t_ft0 = time.perf_counter()
    if TRICOR_FINETUNE_STEPS > 0:
        predicted = _tricor_finetune(
            predicted, species, cell,
            finetune_kwargs, TRICOR_FINETUNE_STEPS,
        )
    t_ft = time.perf_counter() - t_ft0
    dt = t_ml + t_ft

    # Positional RMSE (min-image displacement between predicted and tricor target).
    predicted_t = torch.tensor(predicted, dtype=torch.float64)
    best_t = torch.tensor(best, dtype=torch.float64)
    cell_t = torch.tensor(cell, dtype=torch.float64)
    disp = _min_image_displacement(predicted_t, best_t, cell_t)
    rmse = disp.pow(2).sum(dim=-1).mean().sqrt().item()

    # PDF / ADF match + optional plot.  Skipped entirely when
    # EVAL_METRICS is False so timing runs aren't inflated by the
    # O(N * neighbors^2) ADF computation.
    pdf_mse = float("nan")
    adf_mse = float("nan")
    if EVAL_METRICS:
        g2_pred, adf_pred = _pdf_adf(predicted, species, cell)
        g2_tgt, adf_tgt = _pdf_adf(best, species, cell)
        pdf_mse = float(np.mean((g2_pred - g2_tgt) ** 2))
        adf_mse = float(np.mean((adf_pred - adf_tgt) ** 2))

        if plot_dir is not None:
            if INCLUDE_INITIAL_IN_PLOTS:
                g2_init, adf_init = _pdf_adf(initial, species, cell)
            else:
                g2_init = adf_init = None
            plot_path = plot_dir / (npz_path.stem + ".png")
            title = (
                f"{regime}  ({npz_path.stem})\n"
                f"atoms={predicted.shape[0]}  iters={n_iter}/{MAX_ITER}  "
                f"RMSE={rmse:.3f} Å  PDF MSE={pdf_mse:.2e}  ADF MSE={adf_mse:.2e}"
            )
            _save_comparison_plot(
                plot_path,
                g2_tgt, g2_pred, g2_init,
                adf_tgt, adf_pred, adf_init,
                title,
                species=species,
                cell=cell,
            )

    n_atoms = predicted.shape[0]
    ft_str = (
        f"+{TRICOR_FINETUNE_STEPS}ft={t_ft:4.1f}s "
        if TRICOR_FINETUNE_STEPS > 0 else ""
    )
    metrics_str = (
        f"PDF_MSE={pdf_mse:.3e}  ADF_MSE={adf_mse:.3e}  "
        if EVAL_METRICS else ""
    )
    print(
        f"[{regime:>16s}] atoms={n_atoms:5d}  iters={n_iter:3d}/{MAX_ITER}  "
        f"ml={t_ml:5.2f}s {ft_str}wall={dt:5.2f}s  RMSE={rmse:.4f}Å  "
        f"{metrics_str}"
        f"(tricor best_loss={best_loss:.3f})",
        flush=True,
    )
    return {
        "file": npz_path.name,
        "regime": regime,
        "num_atoms": int(n_atoms),
        "iterations": int(n_iter),
        "wall_time_s": float(dt),
        "rmse_ang": float(rmse),
        "pdf_mse": float(pdf_mse),
        "adf_mse": float(adf_mse),
        "tricor_best_loss": float(best_loss),
    }


def _select_files(target: Path) -> list[Path]:
    """Resolve the configured TARGET into a list of .npz files to evaluate."""
    if target.is_file() and target.suffix == ".npz":
        return [target]
    if not target.is_dir():
        print(f"Not a .npz file or directory: {target}")
        sys.exit(1)

    if SAMPLE_PER_REGIME:
        rng = np.random.default_rng(SAMPLE_SEED)
        picks: list[Path] = []
        for regime in REGIMES:
            candidates = sorted(target.glob(f"si_{regime}_*.npz"))
            if not candidates:
                print(f"  [warn] no files matching si_{regime}_*.npz in {target}")
                continue
            picks.append(candidates[rng.integers(0, len(candidates))])
        return picks
    return sorted(target.glob("*.npz"))


def main() -> None:
    ckpt_path = Path(CHECKPOINT).resolve()
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    target = Path(TARGET).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(ckpt_path, device)
    npz_files = _select_files(target)
    if not npz_files:
        print("No files to evaluate.")
        sys.exit(1)

    if SAVE_PLOTS:
        if PLOT_DIR is None:
            base = target if target.is_dir() else target.parent
            plot_dir = base / "evaluation_plots_tunetest"
        else:
            plot_dir = Path(PLOT_DIR).resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        plot_dir = None

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}  (EMA={USE_EMA_WEIGHTS})")
    mode = "one-per-regime" if (target.is_dir() and SAMPLE_PER_REGIME) else "all"
    print(f"Mode: {mode}  Evaluating {len(npz_files)} file(s)")
    print(f"max_iter={MAX_ITER}  tol={CONVERGENCE_TOL_ANG} Å")
    if plot_dir is not None:
        print(f"Plots: {plot_dir}")
    print()

    results = []
    for p in npz_files:
        try:
            results.append(evaluate_one(model, p, device, plot_dir=plot_dir))
        except Exception as e:
            print(f"  {p.name}: FAILED: {type(e).__name__}: {e}")

    if results:
        rmses = np.array([r["rmse_ang"] for r in results])
        iters = np.array([r["iterations"] for r in results])
        walls = np.array([r["wall_time_s"] for r in results])
        print()
        print(f"N={len(results)}  "
              f"RMSE median={np.median(rmses):.3f} Å  p90={np.percentile(rmses, 90):.3f} Å  "
              f"iters median={np.median(iters):.0f}  "
              f"wall median={np.median(walls):.2f}s")


if __name__ == "__main__":
    main()
