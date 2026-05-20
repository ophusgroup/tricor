"""Iterative inference + comparison for the v2 (per-edge injection) model.

Variant of ``shelltgt_phys/evaluate.py`` that loads a checkpoint trained by
``shelltgt_phys_v2/train.py``.  Per-edge shell_target injection replaces
the per-graph deep-set encoder; data layer and forward signature are
unchanged, so the only difference here is the import.

Edit the CONFIG block below, then run:
    python evaluate.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (resource caps + paths must be set before torch import)
# ─────────────────────────────────────────────────────────────────────────────

# --- buffle resource caps ---
GPU_ID = 0
NUM_THREADS = 2

# --- what to evaluate ---
# Path to a Lightning checkpoint from shelltgt_phys_v2/train.py.
CHECKPOINT = (
    "./lightning_logs/coord_phys_rinject/version_1/checkpoints/last.ckpt"
)

# Path to either a single .npz file or a directory of .npz files.
# For the cross-composition test, point this at the held-out Si3N4
# trajectories (the ones absent from training).
TARGET = "/home/ehrdt/tricor/scripts/relaxml/data/sio2_polymorphs_v1/SiO2/stishovite_trajectories" #"../data/multi_species_v2/Si3N4_trajectories_150/" # "/home/ehrdt/tricor/scripts/relaxml/data/multi_species_v1/Ga2O3_mp-886_trajectories_150" 
#"/home/ehrdt/tricor/scripts/relaxml/data/sio2_polymorphs_v1/SiO2/stishovite_trajectories" # ../data/multi_species_v1/Si3N4_trajectories/"

SAMPLE_PER_REGIME = True
SAMPLE_SEED = 0

REGIMES = (
    "liquid", "amorphous", "SRO", "MRO", "LRO", "nanocrystalline",
)

# --- inference controls ---
MAX_ITER = 50
CONVERGENCE_TOL_ANG = 0.01
USE_EMA_WEIGHTS = True
SPECIES = [14, 7]                # Si, N — informational only; not used to gate
CUTOFF = 5.0

TRICOR_FINETUNE_STEPS = 0

# --- PDF / ADF comparison ---
PDF_R_MAX = 10.0
PDF_R_STEP = 0.05
PDF_PHI_BINS = 90
PLOT_R_MAX = 8.0

EVAL_METRICS = True
# Computing the ADF used to be ~3-5x slower than g(r) alone because the
# inner triplet-routing loop was Python-level; the fast eval path added
# in tricor.differentiable_pdf_fast.compute_eval() vectorizes it via
# g3_lookup gather, so ADF now costs roughly the same as g(r).  Leave
# this True unless you have a reason to skip ADF.
COMPUTE_ADF = True
# "fast"  -> mod.compute_eval()  (vectorized triplet routing + bincount
#                                 histogram, ~50-100x faster than ref)
# "ref"   -> mod.compute()       (reference implementation; slower, gold
#                                 standard for cross-checking ADF shapes
#                                 against the fast path).  Swap to "ref"
#                                 if you suspect the fast eval is
#                                 producing unexpected ADF shapes.
ADF_BACKEND = "fast"
# When True, print per-triplet sum/max of the RAW (un-area-normed) ADF
# for both predicted and target structures right after compute.  Use to
# diagnose whether a triplet's "flat" plot is from low counts (noise +
# normalization) or from a genuine flat distribution.
DEBUG_ADF_PRINT_RAW = False
# Cutoff (Å) for the ADF neighbor search.  ADF is a first-shell quantity
# — angles between *bonded* neighbors of a central atom.  PDF_R_MAX is
# typically too generous for ADF: at 10 Å on a stishovite-density cell
# (0.12 atoms/Å³) each center has ~500 neighbors, and the per-batch
# (B, K_max, K_max) angle tensor OOMs the GPU.  4.0 Å captures the full
# first shell (Si-O ~1.6 Å, M-O ~2.0 Å) with margin and cuts K_max from
# ~500 to ~30 → 250× memory reduction.  Bump to PDF_R_MAX for the old
# behavior if you want long-range angle structure.
ADF_R_MAX = 4.0
# Number of centers processed at once when computing angles.  Default
# 512 in the module is tuned for sparse cells (K_max < 20); on dense
# oxides drop to 64.  Linear memory cost: each batch holds ~10
# (B, K_max, K_max) tensors in flight.
ADF_BATCH_SIZE = 64
PDF_ADF_DEVICE = "auto"
PDF_ADF_DTYPE = "float32"

# --- visualization ---
SAVE_PLOTS = True
PLOT_DIR = None
INCLUDE_INITIAL_IN_PLOTS = True

# Write predicted structures as extended XYZ for visual inspection in
# OVITO / VESTA.  Each .npz produces one .xyz containing 3 frames so you
# can scrub through them: frame 0 = initial (pre-relax),
# frame 1 = ML-predicted, frame 2 = tricor target (best_positions).
# Loads in OVITO as a 3-frame trajectory; the 'frame_label' field in
# each frame's comment line identifies which is which.
SAVE_XYZ = True
XYZ_DIR = None      # None = auto: <TARGET parent>/evaluation_xyz_shelltgt_phys/
# Stride for capturing intermediate ML iterations into the XYZ file.
#   1 = every iteration (~50+ frames per file, animates the full relaxation),
#   5 = every 5th iteration (~10 frames per file),
#   0 = skip intermediates — only write {initial, final, target} (3 frames).
# Final state and tol-convergence state are always included regardless.
XYZ_ITER_STRIDE = 1

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

import numpy as np
import torch
torch.set_num_threads(NUM_THREADS)
from torch_geometric.data import Batch

from tricor.flowmatch.flow_utils import (
    periodic_radius_graph_cell_list,
    periodic_radius_graph_chunked,
)

from tricor.relaxml.shelltgt_phys_v2 import LitRelaxML
from tricor.relaxml.data_shelltgt import (
    ShellTargetData,
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
    if USE_EMA_WEIGHTS and hasattr(lit, "ema_model"):
        lit.ema_model.eval()
        return lit.ema_model.module.to(device)
    return lit.model.to(device)


def _build_data(
    positions: torch.Tensor,
    cell: torch.Tensor,
    z: torch.Tensor,
    weight_vector: torch.Tensor,
    shell: dict,
    cutoff: float,
) -> Batch:
    """Wrap one structure into a single-graph PyG Batch with shell_target."""
    edge_index, edge_vec = _periodic_graph(positions, cutoff, cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])
    data = ShellTargetData(
        z=z,
        pos=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
        w=weight_vector.unsqueeze(0),
        shell_pair_species=shell["pair_species"],
        shell_pair_features=shell["pair_features"],
        shell_pair_batch=torch.zeros(shell["pair_species"].shape[0],
                                     dtype=torch.long, device=positions.device),
        shell_trip_species=shell["trip_species"],
        shell_trip_features=shell["trip_features"],
        shell_trip_batch=torch.zeros(shell["trip_species"].shape[0],
                                     dtype=torch.long, device=positions.device),
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
    shell_target: dict,
    device: torch.device,
    cutoff: float = CUTOFF,
    max_iter: int = MAX_ITER,
    tol: float = CONVERGENCE_TOL_ANG,
    collect_every: int = 0,
) -> tuple[np.ndarray, int, list[tuple[int, np.ndarray]]]:
    pos = torch.tensor(initial_positions, dtype=torch.float32, device=device)
    cell_t = torch.tensor(cell, dtype=torch.float32, device=device)
    w_t = torch.tensor(weight_vector, dtype=torch.float32, device=device)
    z = torch.tensor(species, dtype=torch.long, device=device)

    shell = {
        "pair_species":  torch.tensor(shell_target["shell_pair_species"],
                                      dtype=torch.long, device=device),
        "pair_features": torch.tensor(shell_target["shell_pair_features"],
                                      dtype=torch.float32, device=device),
        "trip_species":  torch.tensor(shell_target["shell_triplet_species"],
                                      dtype=torch.long, device=device),
        "trip_features": torch.tensor(shell_target["shell_triplet_features"],
                                      dtype=torch.float32, device=device),
    }

    # When collect_every > 0, record positions at every Nth iteration so
    # the XYZ writer can animate the relaxation in OVITO.  The final state
    # is always recorded (either via early-convergence return or the
    # post-loop append) so the user never gets a truncated trajectory.
    intermediates: list[tuple[int, np.ndarray]] = []

    def _record(iter_idx: int, p: torch.Tensor) -> None:
        intermediates.append((iter_idx, p.detach().cpu().numpy().copy()))

    for it in range(max_iter):
        batch = _build_data(pos, cell_t, z, w_t, shell, cutoff)
        delta = model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
            batch.shell_pair_species, batch.shell_pair_features,
            batch.shell_pair_batch,
            batch.shell_trip_species, batch.shell_trip_features,
            batch.shell_trip_batch,
        )
        d_norms = delta.norm(dim=-1)
        print(
            f"  iter {it:2d}: |delta| mean={d_norms.mean().item():.4f}  "
            f"max={d_norms.max().item():.4f}  "
            f"std={d_norms.std().item():.4f}  Å"
        )
        pos_new = _wrap_positions(pos + delta, cell_t)
        max_step = (pos_new - pos).norm(dim=-1).max().item()
        pos = pos_new
        if collect_every > 0 and ((it + 1) % collect_every == 0):
            _record(it + 1, pos)
        if max_step < tol:
            # Always record the converged state, even if not on a stride boundary.
            if collect_every > 0 and (not intermediates or intermediates[-1][0] != it + 1):
                _record(it + 1, pos)
            return pos.cpu().numpy(), it + 1, intermediates
    # Hit max_iter without converging — record final state if not already.
    if collect_every > 0 and (not intermediates or intermediates[-1][0] != max_iter):
        _record(max_iter, pos)
    return pos.cpu().numpy(), max_iter, intermediates


def _tricor_finetune(
    positions: np.ndarray,
    species: np.ndarray,
    cell: np.ndarray,
    weights: dict,
    n_steps: int,
) -> np.ndarray:
    """N steps of native shell_relax starting from the ML-predicted state."""
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


# Module cache: DifferentiablePDFADF_Fast is expensive to construct; build
# one per (species, device, dtype) and reuse across structures.
_PDF_ADF_MOD_CACHE: dict[tuple, "torch.nn.Module"] = {}


def _resolve_pdf_adf_device() -> torch.device:
    if PDF_ADF_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(PDF_ADF_DEVICE)


def _resolve_pdf_adf_dtype() -> torch.dtype:
    return torch.float64 if PDF_ADF_DTYPE == "float64" else torch.float32


def _get_pdf_adf_module(species_list: list[int], device: torch.device,
                        dtype: torch.dtype):
    key = (tuple(species_list), str(device), str(dtype))
    mod = _PDF_ADF_MOD_CACHE.get(key)
    if mod is not None:
        return mod
    from tricor.differentiable_pdf_fast import DifferentiablePDFADF_Fast
    mod = DifferentiablePDFADF_Fast(
        r_max=PDF_R_MAX, r_step=PDF_R_STEP,
        phi_num_bins=PDF_PHI_BINS, species=species_list,
        adf_r_max=ADF_R_MAX, adf_batch_size=ADF_BATCH_SIZE,
    ).to(device=device, dtype=dtype)
    _PDF_ADF_MOD_CACHE[key] = mod
    return mod


def _pdf_adf(positions: np.ndarray, species: np.ndarray, cell: np.ndarray):
    species_list = sorted({int(z) for z in species.tolist()})
    device = _resolve_pdf_adf_device()
    dtype = _resolve_pdf_adf_dtype()
    mod = _get_pdf_adf_module(species_list, device, dtype)
    # Re-apply ADF knobs on the cached module so config edits between
    # runs in the same Python session take effect without rebuilding.
    mod.adf_r_max = ADF_R_MAX
    mod.adf_batch_size = ADF_BATCH_SIZE
    pos_t = torch.as_tensor(positions, dtype=dtype, device=device)
    sp_t = torch.as_tensor(species, dtype=torch.int64, device=device)
    cell_t = torch.as_tensor(cell, dtype=dtype, device=device)
    with torch.no_grad():
        if COMPUTE_ADF:
            if ADF_BACKEND == "ref":
                # Reference implementation — slower per-triplet Python
                # loop in tricor.differentiable_pdf_fast.compute().  Use
                # for cross-checking ADF shapes against compute_eval.
                g2, adf = mod.compute(pos_t, sp_t, cell_t)
            else:
                # compute_eval() = vectorized-triplet eval path:
                # numerically identical to compute() but ~5-20x faster
                # on multi-species cells because g3_lookup gathers
                # replace the per-triplet Python loop.  No gradients.
                g2, adf = mod.compute_eval(pos_t, sp_t, cell_t)
        else:
            g2, adf = mod.compute_g2_only(pos_t, sp_t, cell_t)

    if DEBUG_ADF_PRINT_RAW and COMPUTE_ADF:
        # Print sum and peak of the RAW (un-area-normed) ADF for each
        # triplet so a "flat" plot can be diagnosed: if a triplet's sum
        # is orders-of-magnitude smaller than its peers, the flat
        # appearance is normalization noise on near-zero data; if its
        # sum is comparable to the peaked triplets, the flat shape is a
        # real feature of the angle distribution (or a bug worth
        # tracking).
        labels = getattr(mod, "triplet_labels", None)
        adf_cpu = adf.detach().cpu().numpy()
        backend_tag = ADF_BACKEND
        for t in range(adf_cpu.shape[0]):
            label = labels[t] if labels is not None else f"[{t}]"
            print(
                f"      ADF[{backend_tag}] triplet {t} ({label}): "
                f"sum={adf_cpu[t].sum():.3e}  max={adf_cpu[t].max():.3e}"
            )

    return g2.detach().cpu().numpy(), adf.detach().cpu().numpy()


def _pdf_adf_grids() -> tuple[np.ndarray, np.ndarray]:
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
    """One g(r) panel per unique species pair + one ADF panel for all triplets."""
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

    pair_indices = [(i, j)
                    for i in range(len(sorted_species))
                    for j in range(i, len(sorted_species))]
    pair_labels = [
        f"{chemical_symbols[sorted_species[i]]}–{chemical_symbols[sorted_species[j]]}"
        for i, j in pair_indices
    ]
    n_pairs = len(pair_indices)
    n_triplets = adf_tgt.shape[0]

    # Triplet labels (n1-center-n2) reconstructed from the canonical
    # g3_index ordering inside tricor.differentiable_pdf_fast.  Same
    # enumeration: ``c in range(n_species), n1 in range(n_species),
    # n2 in range(n1, n_species)``.  Uses chemical symbols (Si-O-Si)
    # rather than Z numbers (14-8-14) for the subplot titles.
    n_species = len(sorted_species)
    triplet_indices = [
        (c, n1, n2)
        for c in range(n_species)
        for n1 in range(n_species)
        for n2 in range(n1, n_species)
    ]
    assert len(triplet_indices) == n_triplets, (
        f"triplet count mismatch: built {len(triplet_indices)} index entries "
        f"for {n_species} species, but adf array has {n_triplets} rows"
    )
    triplet_labels = [
        f"{chemical_symbols[sorted_species[n1]]}"
        f"–{chemical_symbols[sorted_species[c]]}"
        f"–{chemical_symbols[sorted_species[n2]]}"
        for (c, n1, n2) in triplet_indices
    ]

    # 3-column grid: PDFs in their own row(s), then ADFs in subsequent
    # rows (3 ADF panels per row).  For the common 2-species case
    # (n_pairs=3, n_triplets=6) this is a clean 3×3 grid — top row
    # holds the 3 g(r) panels, lower two rows hold one panel per
    # triplet labeled by chemistry rather than overlaying all 6 on a
    # single axis.
    NCOLS = 3
    import math as _math
    n_pdf_rows = max(1, _math.ceil(n_pairs / NCOLS))
    n_adf_rows = max(1, _math.ceil(n_triplets / NCOLS))
    n_rows = n_pdf_rows + n_adf_rows

    fig, axes = plt.subplots(
        n_rows, NCOLS,
        figsize=(5.5 * NCOLS, 4.0 * n_rows),
        squeeze=False,
    )

    # ─── PDF panels ─────────────────────────────────────────────────
    # Draw order matters when curves coincide: e.g. nanocrystalline
    # initial ≈ target (shell_relax is nearly a no-op on already-good
    # crystals).  Plot target / predicted first, then initial last with
    # the dashed style — dashes alternate with the underlying solid
    # color so both remain visible even when they're identical.
    for k, ((i, j), lbl) in enumerate(zip(pair_indices, pair_labels)):
        ax = axes[k // NCOLS, k % NCOLS]
        ax.plot(r_grid, _gofr(g2_tgt[i, j], i, j),
                color="C3", lw=2.0, label="tricor (target)")
        ax.plot(r_grid, _gofr(g2_pred[i, j], i, j),
                color="C0", lw=1.6, label="model (predicted)")
        if g2_init is not None:
            ax.plot(r_grid, _gofr(g2_init[i, j], i, j), color="0.35", lw=1.2,
                    ls="--", label="initial (pre-relax)")
        ax.set_xlim(0.0, PLOT_R_MAX)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"g(r): {lbl}")
        ax.axhline(1.0, color="0.7", lw=0.7, ls=":")
        ax.legend(framealpha=0.9, fontsize=9)

    # Hide unused PDF slots (when n_pairs isn't a multiple of NCOLS).
    for k in range(n_pairs, n_pdf_rows * NCOLS):
        axes[k // NCOLS, k % NCOLS].set_visible(False)

    # ─── ADF panels: one per triplet ────────────────────────────────
    # Same draw-order rationale as the PDF panels — initial drawn last
    # with dashed style so it remains visible when it coincides with
    # the target distribution.
    for t, tri_lbl in enumerate(triplet_labels):
        row = n_pdf_rows + (t // NCOLS)
        col = t % NCOLS
        ax = axes[row, col]
        ax.plot(phi_grid, _area_norm(adf_tgt[t], phi_grid),
                color="C3", lw=2.0, label="tricor (target)")
        ax.plot(phi_grid, _area_norm(adf_pred[t], phi_grid),
                color="C0", lw=1.6, label="model (predicted)")
        if adf_init is not None:
            ax.plot(phi_grid, _area_norm(adf_init[t], phi_grid),
                    color="0.35", lw=1.2, ls="--", label="initial (pre-relax)")
        ax.set_xlabel("bond angle φ (deg)")
        ax.set_ylabel("ADF(φ)  [normalized]")
        ax.set_title(f"ADF: {tri_lbl}")
        ax.legend(framealpha=0.9, fontsize=9)

    # Hide unused ADF slots (when n_triplets isn't a multiple of NCOLS).
    for t in range(n_triplets, n_adf_rows * NCOLS):
        row = n_pdf_rows + (t // NCOLS)
        col = t % NCOLS
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_comparison_xyz(
    out_path: Path,
    initial: np.ndarray,
    predicted: np.ndarray,
    best: np.ndarray,
    species: np.ndarray,
    cell: np.ndarray,
    regime: str,
    rmse_ang: float,
    intermediates: list[tuple[int, np.ndarray]] | None = None,
) -> None:
    """Write an extended-XYZ animation of the relaxation.

    Frame ordering:
      Frame 0:           initial state (positions[0] from the npz, pre-relax)
      Frames 1..K:       ML iterations (one per entry in ``intermediates``,
                         labelled ``ml_iter_<N>``).  When the inference loop
                         runs with ``collect_every=1`` this is every iter.
                         The last intermediate IS the final ML prediction,
                         so we don't add a separate "predicted" frame.
      Frame K+1 (or 1):  tricor target (best_positions — what the surrogate
                         is trying to reproduce), labelled ``best_tricor``.

    If ``intermediates`` is None or empty, only {initial, predicted,
    best_tricor} are written (the original 3-frame fallback).

    Loads in OVITO as an animation; use the timeline scrubber to step
    through the relaxation.  Each frame's comment line carries
    ``frame_label`` so it's unambiguous which is which.  The cell + PBC
    are written into every frame so OVITO renders the box correctly.
    """
    # Lazy import: keeps module-level imports free of an ase.io dependency
    # for users who only run the script with SAVE_XYZ=False.
    from ase import Atoms
    from ase.io import write

    cell_arr = np.asarray(cell, dtype=np.float64)
    numbers = np.asarray(species, dtype=np.int64)

    def _frame(label: str, pos: np.ndarray, *, iter_idx: int | None = None) -> Atoms:
        atoms = Atoms(
            numbers=numbers,
            positions=np.asarray(pos, dtype=np.float64),
            cell=cell_arr,
            pbc=True,
        )
        atoms.info["frame_label"] = label
        atoms.info["regime"] = regime
        atoms.info["rmse_ang_pred_vs_best"] = float(rmse_ang)
        if iter_idx is not None:
            atoms.info["ml_iter"] = int(iter_idx)
        return atoms

    frames: list[Atoms] = [_frame("initial", initial)]
    if intermediates:
        for it_num, pos in intermediates:
            frames.append(_frame(f"ml_iter_{it_num}", pos, iter_idx=it_num))
    else:
        # No intermediates collected — include the final ML prediction as
        # its own frame so the 3-frame fallback still works.
        frames.append(_frame("predicted", predicted))
    frames.append(_frame("best_tricor", best))

    write(str(out_path), frames, format="extxyz")


def evaluate_one(
    model: torch.nn.Module, npz_path: Path, device: torch.device,
    plot_dir: Path | None = None,
    xyz_dir: Path | None = None,
) -> dict:
    with np.load(npz_path) as npz:
        initial = np.asarray(npz["positions"][0], dtype=np.float32)
        best = np.asarray(npz["best_positions"], dtype=np.float32)
        cell = np.asarray(npz["cell"], dtype=np.float32)
        species = np.asarray(npz["species_numbers"], dtype=np.int64)
        weight_vector = _weight_vector_from_npz(npz)
        regime = str(npz["regime"].item())
        best_loss = float(npz["best_loss"])
        finetune_kwargs = {
            "bond_weight":         float(npz["bond_weight"]),
            "angle_weight":        float(npz["angle_weight"]),
            "repulsion_weight":    float(npz["repulsion_weight"]),
            "hard_core_scale":     float(npz["hard_core_scale"]),
            "nonbond_push_scale":  float(npz["nonbond_push_scale"]),
        }
        if "shell_pair_species" not in npz.files:
            raise KeyError(
                f"{npz_path.name} has no shell_target arrays. Run "
                f"add_shell_target_to_npz.py against the source dir first."
            )
        shell_target = {
            "shell_pair_species":     np.asarray(npz["shell_pair_species"]),
            "shell_pair_features":    np.asarray(npz["shell_pair_features"]),
            "shell_triplet_species":  np.asarray(npz["shell_triplet_species"]),
            "shell_triplet_features": np.asarray(npz["shell_triplet_features"]),
        }

    t0 = time.perf_counter()
    # Collect per-iteration positions when XYZ output is enabled with a
    # nonzero stride.  Disabled otherwise so we don't pay the cpu()-copy
    # cost on pure-metric runs.
    _collect_every = XYZ_ITER_STRIDE if (xyz_dir is not None) else 0
    predicted, n_iter, ml_iters = run_iterative_inference(
        model, initial, cell, species, weight_vector, shell_target, device,
        collect_every=_collect_every,
    )
    t_ml = time.perf_counter() - t0

    t_ft0 = time.perf_counter()
    if TRICOR_FINETUNE_STEPS > 0:
        predicted = _tricor_finetune(
            predicted, species, cell,
            finetune_kwargs, TRICOR_FINETUNE_STEPS,
        )
    t_ft = time.perf_counter() - t_ft0
    dt = t_ml + t_ft

    predicted_t = torch.tensor(predicted, dtype=torch.float64)
    best_t = torch.tensor(best, dtype=torch.float64)
    cell_t = torch.tensor(cell, dtype=torch.float64)
    disp = _min_image_displacement(predicted_t, best_t, cell_t)
    rmse = disp.pow(2).sum(dim=-1).mean().sqrt().item()

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

    # XYZ side-channel: dump the structures themselves so we can flip
    # through them in OVITO.  Independent of EVAL_METRICS so you can use
    # SAVE_XYZ=True even on a pure-timing run.
    if xyz_dir is not None:
        xyz_path = xyz_dir / (npz_path.stem + "_compare.xyz")
        _write_comparison_xyz(
            xyz_path, initial, predicted, best, species, cell,
            regime=regime, rmse_ang=rmse,
            intermediates=ml_iters if ml_iters else None,
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
    """Resolve TARGET into a list of .npz files to evaluate."""
    if target.is_file() and target.suffix == ".npz":
        return [target]
    if not target.is_dir():
        print(f"Not a .npz file or directory: {target}")
        sys.exit(1)

    if SAMPLE_PER_REGIME:
        rng = np.random.default_rng(SAMPLE_SEED)
        picks: list[Path] = []
        for regime in REGIMES:
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
            plot_dir = base / "evaluation_plots_shelltgt_phys_v2"
        else:
            plot_dir = Path(PLOT_DIR).resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        plot_dir = None

    if SAVE_XYZ:
        if XYZ_DIR is None:
            base = target if target.is_dir() else target.parent
            xyz_dir = base / "evaluation_xyz_shelltgt_phys_v2"
        else:
            xyz_dir = Path(XYZ_DIR).resolve()
        xyz_dir.mkdir(parents=True, exist_ok=True)
    else:
        xyz_dir = None

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}  (EMA={USE_EMA_WEIGHTS})")
    mode = "one-per-regime" if (target.is_dir() and SAMPLE_PER_REGIME) else "all"
    print(f"Mode: {mode}  Evaluating {len(npz_files)} file(s)")
    print(f"max_iter={MAX_ITER}  tol={CONVERGENCE_TOL_ANG} Å")
    if plot_dir is not None:
        print(f"Plots: {plot_dir}")
    if xyz_dir is not None:
        print(f"XYZ:   {xyz_dir}")
    print()

    results = []
    for p in npz_files:
        try:
            results.append(evaluate_one(
                model, p, device, plot_dir=plot_dir, xyz_dir=xyz_dir,
            ))
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
