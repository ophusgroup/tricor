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
CHECKPOINT = "./lightning_logs/relaxml-subset-multi-si-real/version_0/checkpoints/last.ckpt"

# Path to either a single .npz file or a directory of .npz files.
TARGET = "./data/multi_species_v1/SiO2_trajectories" #"./data/si_trajectories_v2"

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
CUTOFF = 5.0

# --- optional tricor "finetune" after ML inference ---
# Run this many additional shell_relax steps with the original weight
# parameters after the iterative ML inference converges.  0 disables.
# 5–20 is the useful range — enough to clean up small residual errors
# at the disorder extremes without losing the wall-time win.  Uses the
# weight params stored in the .npz (so behavior matches what tricor
# would have done from the start, just on a much-better starting state).
TRICOR_FINETUNE_STEPS = 0

# --- PDF / ADF comparison ---
# PDF_R_MAX is computed slightly past the 8 Å plot crop so that the
# Gaussian-tail clipping artifact at the neighbor-list cutoff edge stays
# off-screen.  Bump together if PLOT_R_MAX changes.
PDF_R_MAX = 9.0
PDF_R_STEP = 0.05
PDF_PHI_BINS = 90
PLOT_R_MAX = 8.0                 # x-axis limit for the g(r) plot panel

# Set False to skip the PDF/ADF metric computation entirely (also disables
# plotting, since plots reuse those arrays).  Useful for pure timing runs.
EVAL_METRICS = True

# PDF/ADF compute speed knobs.  ADF is the slow part (triplets scale as
# O(N * neighbors^2)).  Set COMPUTE_ADF=False for ~3-5x speedup when you
# only care about g(r); the ADF panel/MSE will be zeros.  PDF_ADF_DEVICE
# auto-picks GPU when available (much faster than CPU at large N).
COMPUTE_ADF = False
PDF_ADF_DEVICE = "auto"          # "auto" | "cuda" | "cpu"
PDF_ADF_DTYPE = "float32"        # "float32" or "float64"

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
from torch_geometric.data import Data, Batch

from tricor.flowmatch.flow_utils import (
    periodic_radius_graph_cell_list,
    periodic_radius_graph_chunked,
)

from tricor.relaxml import LitRelaxML
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
    # Strip the "._orig_mod." prefix that torch.compile injects into every
    # state-dict key (train.py wraps lit.model in torch.compile before
    # fitting).  Catches both `model._orig_mod.…` and
    # `ema_model.module._orig_mod.…` in one substitution.
    state = torch.load(str(ckpt_path), map_location=device)
    sd = {k.replace("._orig_mod.", "."): v for k, v in state["state_dict"].items()}
    lit = LitRelaxML(**state["hyper_parameters"])
    lit.load_state_dict(sd, strict=True)
    lit.eval().to(device)
    # Pull the EMA weights into the main model if requested — these are
    # generally what you want for inference.
    if USE_EMA_WEIGHTS and hasattr(lit, "ema_model"):
        lit.ema_model.eval()
        return lit.ema_model.module.to(device)
    return lit.model.to(device)


def _build_data(
    positions: torch.Tensor,
    cell: torch.Tensor,
    z: torch.Tensor,
    weight_vector: torch.Tensor,
    cutoff: float,
) -> Batch:
    edge_index, edge_vec = _periodic_graph(positions, cutoff, cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])
    data = Data(
        z=z,                                 # (N,) long atomic numbers
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
    z = torch.tensor(species, dtype=torch.long, device=device)

    for it in range(max_iter):
        batch = _build_data(pos, cell_t, z, w_t, cutoff)
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

    # Si-only path.  For multi-species we'd need to derive the right
    # reference cell from the .npz (compound formula + lattice params).
    # Until that's in place, refuse to silently produce wrong results.
    unique_z = sorted({int(z) for z in np.asarray(species).tolist()})
    if unique_z != [14]:
        raise NotImplementedError(
            f"_tricor_finetune currently only supports Si (Z=14); got species "
            f"{unique_z}.  Set TRICOR_FINETUNE_STEPS = 0 for non-Si data."
        )

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


# Module cache: DifferentiablePDFADF_Fast is expensive to construct and
# its parameters don't depend on the per-structure positions, so we
# build one per (species, device, dtype) combination and reuse.
_PDF_ADF_MOD_CACHE: dict[tuple, "torch.nn.Module"] = {}


def _resolve_pdf_adf_device() -> torch.device:
    if PDF_ADF_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(PDF_ADF_DEVICE)


def _resolve_pdf_adf_dtype() -> torch.dtype:
    return torch.float64 if PDF_ADF_DTYPE == "float64" else torch.float32


def _get_pdf_adf_module(species_list: list[int], device: torch.device,
                        dtype: torch.dtype):
    """Return a cached DifferentiablePDFADF_Fast for this species set."""
    key = (tuple(species_list), str(device), str(dtype))
    mod = _PDF_ADF_MOD_CACHE.get(key)
    if mod is not None:
        return mod
    from tricor.differentiable_pdf_fast import DifferentiablePDFADF_Fast
    mod = DifferentiablePDFADF_Fast(
        r_max=PDF_R_MAX, r_step=PDF_R_STEP,
        phi_num_bins=PDF_PHI_BINS, species=species_list,
    ).to(device=device, dtype=dtype)
    _PDF_ADF_MOD_CACHE[key] = mod
    return mod


def _pdf_adf(positions: np.ndarray, species: np.ndarray, cell: np.ndarray):
    """Compute g2 (+ optionally ADF) for a single structure.

    Speed knobs from the CONFIG block:
      - PDF_ADF_DEVICE: "auto" / "cuda" / "cpu"
      - PDF_ADF_DTYPE:  "float32" / "float64"
      - COMPUTE_ADF:    when False, skips the expensive triplet sum and
                        returns a zero ADF tensor for API compatibility.
    Species list is derived from the structure itself so the same evaluator
    works for any 1+-element compound.
    """
    species_list = sorted({int(z) for z in species.tolist()})
    device = _resolve_pdf_adf_device()
    dtype = _resolve_pdf_adf_dtype()
    mod = _get_pdf_adf_module(species_list, device, dtype)
    pos_t = torch.as_tensor(positions, dtype=dtype, device=device)
    sp_t = torch.as_tensor(species, dtype=torch.int64, device=device)
    cell_t = torch.as_tensor(cell, dtype=dtype, device=device)
    with torch.no_grad():
        if COMPUTE_ADF:
            g2, adf = mod.compute(pos_t, sp_t, cell_t)
        else:
            g2, adf = mod.compute_g2_only(pos_t, sp_t, cell_t)
    return g2.detach().cpu().numpy(), adf.detach().cpu().numpy()


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
    """One g(r) panel per unique species pair + one ADF panel for all triplets.

    g(r) uses the standard pair-correlation normalization:
        g_αβ(r) = count_αβ(r) * V / (N_α * (N_β - δ_αβ) * 4π r² * dr)
    so a uniform random arrangement gives g(r) → 1 at large r.

    ADF stays area-normalized (probability density over φ); when there are
    multiple triplet types they're overlaid on the same axes with a legend.
    """
    # Lazy import + Agg backend so headless mallard sessions never need a display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.data import chemical_symbols

    r_grid, phi_grid = _pdf_adf_grids()

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

    cell_arr = np.asarray(cell, dtype=np.float64)
    V = float(abs(np.linalg.det(cell_arr)))
    species_arr = np.asarray(species, dtype=np.int64)
    sorted_species = sorted({int(z) for z in species_arr.tolist()})
    n_per = np.array([(species_arr == Z).sum() for Z in sorted_species],
                     dtype=np.float64)
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

    # Unique pairs (i ≤ j) and their human-readable labels.
    pair_indices = [(i, j)
                    for i in range(len(sorted_species))
                    for j in range(i, len(sorted_species))]
    pair_labels = [
        f"{chemical_symbols[sorted_species[i]]}–{chemical_symbols[sorted_species[j]]}"
        for i, j in pair_indices
    ]
    n_pairs = len(pair_indices)
    n_triplets = adf_tgt.shape[0]

    # Layout: one row, (n_pairs g(r) panels) + (1 ADF panel).
    fig, axes = plt.subplots(
        1, n_pairs + 1,
        figsize=(5.5 * (n_pairs + 1), 4.5),
        squeeze=False,
    )
    axes = axes[0]

    for k, ((i, j), lbl) in enumerate(zip(pair_indices, pair_labels)):
        ax = axes[k]
        if g2_init is not None:
            ax.plot(r_grid, _gofr(g2_init[i, j], i, j), color="0.65", lw=1.0,
                    ls="--", label="initial (pre-relax)")
        ax.plot(r_grid, _gofr(g2_tgt[i, j], i, j),
                color="C3", lw=2.0, label="tricor (target)")
        ax.plot(r_grid, _gofr(g2_pred[i, j], i, j),
                color="C0", lw=1.6, label="model (predicted)")
        ax.set_xlim(0.0, PLOT_R_MAX)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"g(r): {lbl}")
        ax.axhline(1.0, color="0.7", lw=0.7, ls=":")
        ax.legend(framealpha=0.9, fontsize=9)

    ax = axes[-1]
    # Multi-triplet ADF: each triplet type gets its own line trio
    # (target solid, predicted dashed-thinner, initial dotted).  For the
    # common single-triplet case (Si-only) this collapses to the simple
    # 2/3-line plot of before.
    cmap = plt.get_cmap("tab10")
    for t in range(n_triplets):
        c = cmap(t % 10)
        if adf_init is not None:
            ax.plot(phi_grid, _area_norm(adf_init[t], phi_grid),
                    color=c, lw=0.8, ls=":", alpha=0.6)
        ax.plot(phi_grid, _area_norm(adf_tgt[t], phi_grid),
                color=c, lw=2.0,
                label=(f"tricor [{t}]" if n_triplets > 1 else "tricor (target)"))
        ax.plot(phi_grid, _area_norm(adf_pred[t], phi_grid),
                color=c, lw=1.4, ls="--",
                label=(f"model [{t}]" if n_triplets > 1 else "model (predicted)"))
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
            # Compound-agnostic match: filenames are
            # "<formula>_<regime>_cell###_idx#####_seed#########.npz".
            candidates = sorted(target.glob(f"*_{regime}_*.npz"))
            if not candidates:
                print(f"  [warn] no files matching *_{regime}_*.npz in {target}")
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
            plot_dir = base / "evaluation_plots"
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
