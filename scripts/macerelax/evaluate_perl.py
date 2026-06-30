"""Iterative inference + comparison against MACE+wall ground truth — Perlmutter version.

Adapted from the buffle evaluate_perl.py.  Same overall flow, but updated
for the pure-PyTorch DDP training stack:

  * Loads ``.pt`` checkpoints (saved by ``train_perl_ddp.py``'s
    ``RelaxMLTrainer._save_checkpoint``) instead of Lightning ``.ckpt`` files.
  * Walks ``lightning_logs/<RUN_NAME>/run_<timestamp>/checkpoints/`` instead of
    the old ``version_N/`` layout.
  * Builds a bare ``RelaxMLModel`` (no LitRelaxML), then loads either the
    main weights or the EMA state from the checkpoint payload.
  * Architecture hyperparameters (NODE_DIM, NUM_CONVS, …) are pinned in this
    file's CONFIG block — they MUST match what ``train_perl_ddp.py`` used at
    training time.  See the "MODEL ARCHITECTURE" subsection.

Run on Perlmutter:
    /global/common/software/m5020/ehrdt/tricor/bin/python \\
        scripts/macerelax/evaluate_perl.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (resource caps must be set before torch import)
# ─────────────────────────────────────────────────────────────────────────────

# --- resource caps ---
GPU_ID = 0
NUM_THREADS = 4

# --- what to evaluate ---
EXPERIMENT_NAME = "ddp_v1"               # matches RUN_NAME in train_perl_ddp.py
# Pick a specific run_<timestamp> subdir name (e.g. "run_1781157517") or None
# to auto-pick the most recently modified one.
RUN_TIMESTAMP = None
# "last" | "best" | "<absolute path to .pt>"
EPOCH = "best"
# True → load EMA weights into the model (recommended for eval).
USE_EMA_WEIGHTS = True

# Override the eval data source.  None = use the experiment's eval_manifest
# from the EXPERIMENTS_REGISTRY (or fallback to experiment.json saved by
# train_perl_ddp.py's _write_metadata).
TARGET_OVERRIDE = None

# Filter the eval manifest to specific system_ids (None = use all).
SYSTEMS_FILTER: tuple[str, ...] | None = None

# --- Perlmutter paths ---
LOG_DIR              = "/pscratch/sd/e/ehrdt/macerelax/lightning_logs"
EXPERIMENTS_REGISTRY = "/pscratch/sd/e/ehrdt/tricor/cnos_1e100meV/manifests/experiments.json"

# ─── MODEL ARCHITECTURE ─────────────────────────────────────────────────────
# These MUST match what train_perl_ddp.py used to produce the checkpoint.
# The training script's CONFIG block is the source of truth — if you change
# any of these there, change them here.
MAX_Z                    = 120
NODE_DIM                 = 128
EDGE_DIM                 = 128
NUM_CONVS                = 4
WEIGHT_ENCODER_HIDDEN    = 64
SPECIES_PAIR_DIM         = 16
SHELL_TARGET_SPECIES_DIM = 8
SHELL_TARGET_HIDDEN      = 64
SHELL_TARGET_DROPOUT     = 0.0

# --- inference controls ---
MAX_ITER             = None       # None → auto from k_stride + TRAIN_N_STEPS
TRAIN_N_STEPS        = 60         # default in train_perl_ddp.py production runs
MAX_ITER_BUFFER      = 0
CONVERGENCE_TOL_ANG  = 0.001
CUTOFF               = 5.0

# --- PDF / ADF comparison ---
PDF_R_MAX     = 10.0
PDF_R_STEP    = 0.05
PDF_PHI_BINS  = 90
PLOT_R_MAX    = 8.0
ADF_R_MAX     = 2.2
EVAL_METRICS  = True
COMPUTE_ADF   = True
PDF_ADF_DEVICE = "auto"
PDF_ADF_DTYPE  = "float32"

# --- visualization ---
SAVE_PLOTS = True
PLOT_DIR   = None
INCLUDE_INITIAL_IN_PLOTS = True
SAVE_XYZ   = True
XYZ_DIR    = None
XYZ_ITER_STRIDE = 1

# ─────────────────────────────────────────────────────────────────────────────

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_ID))
_n = str(NUM_THREADS)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _n)

import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
torch.set_num_threads(NUM_THREADS)
from torch_geometric.data import Batch

from tricor.macerelax.flow_utils import (
    periodic_radius_graph_cell_list,
    periodic_radius_graph_chunked,
)
from tricor.macerelax.model import RelaxMLModel
from tricor.macerelax.data import (
    ShellTargetData,
    _min_image_displacement,
    _weight_vector_from_row,
)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment / run / checkpoint resolution (pure-PyTorch layout)
# ──────────────────────────────────────────────────────────────────────────────


def _load_experiment(experiment_name: str, registry_path: str,
                      fallback_dir: Path | None = None) -> dict:
    """Resolve an experiment's metadata.

    Tries the registry first.  Falls back to the per-run experiment.json
    that train_perl_ddp.py writes at run-start.
    """
    try:
        with open(registry_path) as f:
            reg = json.load(f)
        exps = reg.get("experiments", {})
    except (FileNotFoundError, json.JSONDecodeError):
        exps = {}

    if experiment_name in exps:
        e = exps[experiment_name]
        base = Path(registry_path).parent
        return {
            **e,
            "experiment_name": experiment_name,
            "train_manifest": str((base / e["train_manifest"]).resolve()),
            "eval_manifest":  str((base / e["eval_manifest"]).resolve()),
        }

    if fallback_dir is None:
        raise SystemExit(
            f"[abort] experiment {experiment_name!r} not in registry "
            f"{registry_path} (available: {sorted(exps)}) and no "
            f"fallback dir was provided."
        )
    candidate = Path(fallback_dir) / "experiment.json"
    if not candidate.is_file():
        raise SystemExit(
            f"[abort] experiment {experiment_name!r} not in registry "
            f"{registry_path} (available: {sorted(exps)}), and no "
            f"experiment.json found at {candidate}."
        )
    with open(candidate) as f:
        d = json.load(f)
    return {
        "experiment_name":       d.get("experiment_name", experiment_name),
        "description":           d.get("experiment_desc",
                                        d.get("description", "")),
        "train_manifest":        d.get("manifest", ""),
        "eval_manifest":         d["eval_manifest"],
        "train_systems":         d.get("train_systems", []),
        "eval_systems":          d.get("eval_systems", []),
        "n_train_trajectories":  int(d.get("n_train_trajectories", 0)),
        "n_eval_trajectories":   int(d.get("n_eval_trajectories", 0)),
        "k_stride_snapshots":    int(d.get("k_stride_snapshots", 1)),
        "cutoff":                float(d.get("cutoff", CUTOFF)),
        "rotate":                bool(d.get("rotate", True)),
    }


def _resolve_run_dir(log_dir: str, experiment_name: str,
                     run_timestamp: str | None) -> Path:
    """Resolve LOG_DIR/EXPERIMENT_NAME/run_<timestamp>/.

    If run_timestamp is None, picks the most recently modified run_* subdir.
    """
    exp_dir = Path(log_dir) / experiment_name
    if not exp_dir.is_dir():
        raise SystemExit(
            f"[abort] no training logs at {exp_dir}.  Did training run?"
        )
    if run_timestamp is not None:
        # Accept either "run_1781157517" or "1781157517".
        name = run_timestamp if run_timestamp.startswith("run_") \
                else f"run_{run_timestamp}"
        run = exp_dir / name
        if not run.is_dir():
            raise SystemExit(f"[abort] {run} not found")
        return run
    candidates = sorted(
        (d for d in exp_dir.iterdir() if d.name.startswith("run_") and d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"[abort] no run_* dirs under {exp_dir}")
    return candidates[-1]


def _resolve_checkpoint(run_dir: Path, epoch: str) -> Path:
    """Find the .pt to load.

    epoch == "last" → run_dir/checkpoints/last.pt
    epoch == "best" → run_dir/checkpoints/best.pt
    Otherwise treat epoch as an absolute path to a .pt file.
    """
    if epoch not in ("last", "best") and Path(epoch).is_file():
        return Path(epoch).resolve()
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"[abort] no checkpoints/ under {run_dir}")
    target = ckpt_dir / f"{epoch}.pt"
    if not target.is_file():
        # Fall back across the alternatives if the user-requested one
        # is missing — e.g. early in training there's no best yet.
        alt = "last" if epoch == "best" else "best"
        alt_path = ckpt_dir / f"{alt}.pt"
        if alt_path.is_file():
            print(f"[warn] {target.name} not found, falling back to {alt}.pt")
            return alt_path.resolve()
        raise SystemExit(f"[abort] no {epoch}.pt (or alternative) in {ckpt_dir}")
    return target.resolve()


def _load_eval_rows(eval_manifest: str,
                    systems_filter: tuple[str, ...] | None) -> list[dict]:
    rows: list[dict] = []
    with open(eval_manifest, newline="") as f:
        reader = csv.DictReader(f)
        manifest_dir = Path(eval_manifest).parent
        for r in reader:
            if systems_filter and r["system_id"] not in systems_filter:
                continue
            r["_npz_path"] = (manifest_dir / r["filename"]).resolve()
            rows.append(r)
    return rows


def _files_from_target_override(target: Path) -> list[dict]:
    paths: list[Path] = []
    if target.is_file() and target.suffix == ".npz":
        paths = [target]
    elif target.is_dir():
        paths = sorted(target.rglob("*.npz"))
    else:
        raise SystemExit(f"[abort] TARGET_OVERRIDE not a .npz or dir: {target}")
    rows: list[dict] = []
    for p in paths:
        with np.load(p) as d:
            sid = str(d["system_id"].item()) if "system_id" in d.files else "?"
            reg = str(d["regime"].item()) if "regime" in d.files else "?"
        rows.append({
            "system_id": sid, "regime": reg, "_npz_path": p,
            "filename": p.name,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Model loading — .pt checkpoint format from train_perl_ddp.py
# ──────────────────────────────────────────────────────────────────────────────


def _build_model() -> RelaxMLModel:
    """Construct a fresh RelaxMLModel with the architecture pinned in CONFIG."""
    return RelaxMLModel(
        max_z=MAX_Z,
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        num_convs=NUM_CONVS,
        weight_encoder_hidden=WEIGHT_ENCODER_HIDDEN,
        species_pair_dim=SPECIES_PAIR_DIM,
        shell_target_species_dim=SHELL_TARGET_SPECIES_DIM,
        shell_target_hidden=SHELL_TARGET_HIDDEN,
        shell_target_dropout=SHELL_TARGET_DROPOUT,
    )


def _load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load weights from a train_perl_ddp.py .pt checkpoint.

    The .pt payload is::

        {
            "model":      raw_model.state_dict(),   # unprefixed keys
            "ema":        ema_state,                # dict[str, Tensor]
            "optimizer":  ...,
            "scheduler":  ...,
            "epoch":      int,
            "best_val":   float,
            "no_improve": int,
        }

    If USE_EMA_WEIGHTS, load the EMA state into the model; otherwise the
    plain "model" state.  Both share the same shape — EMA is a tracking
    copy of the same parameters.
    """
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    model = _build_model()
    # Freeze the edge_norms[-1] layer — must match training to load cleanly.
    for p in model.processor.edge_norms[-1].parameters():
        p.requires_grad_(False)

    # Pick which state dict to load.  Strip "_orig_mod." prefixes that might
    # leak in if a future training run saved compiled state directly.
    src = payload["ema"] if USE_EMA_WEIGHTS else payload["model"]
    src = {k.removeprefix("_orig_mod."): v for k, v in src.items()}

    missing, unexpected = model.load_state_dict(src, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  first missing : {missing[:3]}")
        if unexpected:
            print(f"  first unexpected: {unexpected[:3]}")

    epoch = int(payload.get("epoch", -1))
    best_val = float(payload.get("best_val", float("nan")))
    print(f"[ckpt] loaded {ckpt_path.name}: epoch={epoch}  "
          f"best_val={best_val:.4e}  weights={'EMA' if USE_EMA_WEIGHTS else 'live'}")

    model.eval().to(device)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Iterative inference (unchanged from the prior version)
# ──────────────────────────────────────────────────────────────────────────────


def _periodic_graph(pos, cutoff, cell):
    try:
        return periodic_radius_graph_cell_list(pos, cutoff, cell)
    except ValueError:
        return periodic_radius_graph_chunked(pos, cutoff, cell=cell)


def _build_data(positions, cell, z, weight_vector, shell, cutoff) -> Batch:
    edge_index, edge_vec = _periodic_graph(positions, cutoff, cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])
    data = ShellTargetData(
        z=z, pos=positions, edge_index=edge_index, edge_attr=edge_attr,
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
    fake_row = {
        "grain_size":           str(float(npz["grain_size"])),
        "num_grains":           str(int(npz["num_grains"])),
        "crystalline_fraction": str(float(npz["crystalline_fraction"])),
        "rel_density":          str(float(npz["rel_density"])),
        "wall_global_min":      str(float(npz["wall_global_min"])),
        "fmax_initial":         str(float(npz["fmax_initial"])),
    }
    return _weight_vector_from_row(fake_row)


def _wrap_positions(pos, cell):
    inv_cell = torch.linalg.inv(cell)
    frac = pos @ inv_cell.T
    frac = frac - torch.floor(frac)
    return frac @ cell


@torch.no_grad()
def run_iterative_inference(
    model, initial_positions, cell, species, weight_vector, shell_target,
    device, *, cutoff=CUTOFF, max_iter=None, tol=CONVERGENCE_TOL_ANG,
    collect_every=0,
):
    if max_iter is None:
        max_iter = MAX_ITER
    if max_iter is None:
        raise RuntimeError("MAX_ITER not set — main() should resolve it.")
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
            if collect_every > 0 and (not intermediates or intermediates[-1][0] != it + 1):
                _record(it + 1, pos)
            return pos.cpu().numpy(), it + 1, intermediates
    if collect_every > 0 and (not intermediates or intermediates[-1][0] != max_iter):
        _record(max_iter, pos)
    return pos.cpu().numpy(), max_iter, intermediates


# ──────────────────────────────────────────────────────────────────────────────
# PDF / ADF metrics + plotting (unchanged)
# ──────────────────────────────────────────────────────────────────────────────


_PDF_ADF_MOD_CACHE: dict[tuple, "torch.nn.Module"] = {}


def _resolve_pdf_adf_device():
    if PDF_ADF_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(PDF_ADF_DEVICE)


def _resolve_pdf_adf_dtype():
    return torch.float64 if PDF_ADF_DTYPE == "float64" else torch.float32


def _get_pdf_adf_module(species_list, device, dtype):
    key = (tuple(species_list), str(device), str(dtype))
    mod = _PDF_ADF_MOD_CACHE.get(key)
    if mod is not None:
        return mod
    from tricor.differentiable_pdf_fast import DifferentiablePDFADF_Fast
    mod = DifferentiablePDFADF_Fast(
        r_max=PDF_R_MAX, r_step=PDF_R_STEP,
        phi_num_bins=PDF_PHI_BINS, species=species_list,
        adf_r_max=ADF_R_MAX,
    ).to(device=device, dtype=dtype)
    _PDF_ADF_MOD_CACHE[key] = mod
    return mod


def _pdf_adf(positions, species, cell):
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


def _pdf_adf_grids():
    num_r = int(round(PDF_R_MAX / PDF_R_STEP))
    r_grid = np.arange(num_r, dtype=np.float64) * PDF_R_STEP + 0.5 * PDF_R_STEP
    phi_edges = np.linspace(0.0, np.pi, PDF_PHI_BINS + 1)
    phi_centers = phi_edges[:-1] + 0.5 * (phi_edges[1] - phi_edges[0])
    return r_grid, np.rad2deg(phi_centers)


def _save_comparison_plot(out_path, g2_tgt, g2_pred, g2_init,
                           adf_tgt, adf_pred, adf_init, title, species, cell):
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

    def _gofr(g_pair, i, j):
        ni, nj = n_per[i], n_per[j]
        denom = ni * (ni - 1) if i == j else ni * nj
        if denom <= 0:
            return g_pair * 0.0
        return g_pair * V / (denom * shell)

    def _area_norm(y, x):
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
    n_species = len(sorted_species)
    n_triplets = adf_tgt.shape[0]
    triplets_per_center = n_triplets // n_species

    ncols = max(n_pairs, n_species)
    fig, axes = plt.subplots(
        2, ncols, figsize=(5.5 * ncols, 9.0), squeeze=False,
    )

    for k, ((i, j), lbl) in enumerate(zip(pair_indices, pair_labels)):
        ax = axes[0, k]
        ax.plot(r_grid, _gofr(g2_tgt[i, j], i, j),
                color="C3", lw=2.0, label="MACE+wall (target)")
        ax.plot(r_grid, _gofr(g2_pred[i, j], i, j),
                color="C0", lw=1.6, label="model (predicted)")
        if g2_init is not None:
            ax.plot(r_grid, _gofr(g2_init[i, j], i, j),
                    color="0.2", lw=1.3, ls=(0, (4, 2)), alpha=0.9,
                    zorder=5, label="initial (pre-relax)")
        ax.set_xlim(0.0, PLOT_R_MAX)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"g(r): {lbl}")
        ax.axhline(1.0, color="0.7", lw=0.7, ls=":")
        ax.legend(framealpha=0.9, fontsize=9)
    for k in range(n_pairs, ncols):
        axes[0, k].axis("off")

    for c in range(n_species):
        ax = axes[1, c]
        center_sym = chemical_symbols[sorted_species[c]]
        idxs = list(range(c * triplets_per_center,
                          (c + 1) * triplets_per_center))
        tgt_sum  = adf_tgt[idxs].sum(axis=0)
        pred_sum = adf_pred[idxs].sum(axis=0)
        ax.plot(phi_grid, _area_norm(tgt_sum,  phi_grid),
                color="C3", lw=2.0, label="MACE+wall (target)")
        ax.plot(phi_grid, _area_norm(pred_sum, phi_grid),
                color="C0", lw=1.6, label="model (predicted)")
        if adf_init is not None:
            init_sum = adf_init[idxs].sum(axis=0)
            ax.plot(phi_grid, _area_norm(init_sum, phi_grid),
                    color="0.2", lw=1.3, ls=(0, (4, 2)), alpha=0.9,
                    zorder=5, label="initial (pre-relax)")
        ax.set_xlabel("bond angle φ (deg)")
        ax.set_ylabel("ADF(φ) [normalized]")
        ax.set_title(f"ADF: {center_sym}-centered")
        ax.legend(framealpha=0.9, fontsize=9)
    for c in range(n_species, ncols):
        axes[1, c].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_comparison_xyz(out_path, initial, predicted, best, species, cell,
                           regime, rmse_ang, intermediates=None):
    from ase import Atoms
    from ase.io import write

    cell_arr = np.asarray(cell, dtype=np.float64)
    numbers = np.asarray(species, dtype=np.int64)

    def _frame(label, pos, *, iter_idx=None):
        atoms = Atoms(numbers=numbers, positions=np.asarray(pos, dtype=np.float64),
                      cell=cell_arr, pbc=True)
        atoms.info["frame_label"] = label
        atoms.info["regime"] = regime
        atoms.info["rmse_ang_pred_vs_best"] = float(rmse_ang)
        if iter_idx is not None:
            atoms.info["ml_iter"] = int(iter_idx)
        return atoms

    frames = [_frame("initial", initial)]
    if intermediates:
        for it_num, pos in intermediates:
            frames.append(_frame(f"ml_iter_{it_num}", pos, iter_idx=it_num))
    else:
        frames.append(_frame("predicted", predicted))
    frames.append(_frame("best_mace_wall", best))

    write(str(out_path), frames, format="extxyz")


# ──────────────────────────────────────────────────────────────────────────────
# Per-NPZ evaluation
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_one(model, npz_path: Path, device,
                  plot_dir: Path | None = None,
                  xyz_dir: Path | None = None) -> dict:
    with np.load(npz_path) as npz:
        initial = np.asarray(npz["positions"][0], dtype=np.float32)
        best = np.asarray(npz["best_positions"], dtype=np.float32)
        cell = np.asarray(npz["cell"], dtype=np.float32)
        species = np.asarray(npz["species_numbers"], dtype=np.int64)
        weight_vector = _weight_vector_from_npz(npz)
        regime = str(npz["regime"].item())
        system_id = str(npz["system_id"].item()) if "system_id" in npz.files else "?"
        best_loss = float(npz["best_loss"])
        if "shell_pair_species" not in npz.files:
            raise KeyError(
                f"{npz_path.name} has no shell_target arrays."
            )
        shell_target = {
            "shell_pair_species":     np.asarray(npz["shell_pair_species"]),
            "shell_pair_features":    np.asarray(npz["shell_pair_features"]),
            "shell_triplet_species":  np.asarray(npz["shell_triplet_species"]),
            "shell_triplet_features": np.asarray(npz["shell_triplet_features"]),
        }

    t0 = time.perf_counter()
    collect_every = XYZ_ITER_STRIDE if (xyz_dir is not None) else 0
    predicted, n_iter, ml_iters = run_iterative_inference(
        model, initial, cell, species, weight_vector, shell_target, device,
        collect_every=collect_every,
    )
    dt = time.perf_counter() - t0

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
                f"{system_id} / {regime}  ({npz_path.stem})\n"
                f"atoms={predicted.shape[0]}  iters={n_iter}/{MAX_ITER}  "
                f"RMSE={rmse:.3f} Å  PDF MSE={pdf_mse:.2e}  ADF MSE={adf_mse:.2e}"
            )
            _save_comparison_plot(
                plot_path, g2_tgt, g2_pred, g2_init,
                adf_tgt, adf_pred, adf_init, title,
                species=species, cell=cell,
            )

    if xyz_dir is not None:
        xyz_path = xyz_dir / (npz_path.stem + "_compare.xyz")
        _write_comparison_xyz(
            xyz_path, initial, predicted, best, species, cell,
            regime=regime, rmse_ang=rmse,
            intermediates=ml_iters if ml_iters else None,
        )

    n_atoms = predicted.shape[0]
    metrics_str = (
        f"PDF_MSE={pdf_mse:.3e}  ADF_MSE={adf_mse:.3e}  "
        if EVAL_METRICS else ""
    )
    print(
        f"[{system_id:>18s} / {regime:>16s}] atoms={n_atoms:5d}  "
        f"iters={n_iter:3d}/{MAX_ITER}  ml={dt:5.2f}s  "
        f"RMSE={rmse:.4f}Å  {metrics_str}"
        f"(MACE+wall best_loss={best_loss:.3f})",
        flush=True,
    )
    return {
        "file":              npz_path.name,
        "system_id":         system_id,
        "regime":            regime,
        "num_atoms":         int(n_atoms),
        "iterations":        int(n_iter),
        "wall_time_s":       float(dt),
        "rmse_ang":          float(rmse),
        "pdf_mse":           float(pdf_mse),
        "adf_mse":           float(adf_mse),
        "macewall_best_loss": float(best_loss),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────


def _summarize(results: list[dict]) -> None:
    if not results:
        return
    rmses = np.array([r["rmse_ang"] for r in results])
    iters = np.array([r["iterations"] for r in results])
    walls = np.array([r["wall_time_s"] for r in results])
    print()
    print(f"OVERALL  N={len(results):d}  "
          f"RMSE median={np.median(rmses):.3f} Å  "
          f"p90={np.percentile(rmses, 90):.3f} Å  "
          f"iters median={np.median(iters):.0f}  "
          f"wall median={np.median(walls):.2f}s")

    by_sys: dict[str, list[dict]] = {}
    for r in results:
        by_sys.setdefault(r["system_id"], []).append(r)
    print("\nBy system_id:")
    print(f"  {'system_id':>22s}  {'N':>3s}  {'RMSE med':>9s}  "
          f"{'PDF med':>10s}  {'ADF med':>10s}  {'iters med':>10s}")
    for sid in sorted(by_sys):
        rs = by_sys[sid]
        r_arr = np.array([x["rmse_ang"] for x in rs])
        p_arr = np.array([x["pdf_mse"] for x in rs])
        a_arr = np.array([x["adf_mse"] for x in rs])
        i_arr = np.array([x["iterations"] for x in rs])
        print(f"  {sid:>22s}  {len(rs):>3d}  "
              f"{np.median(r_arr):>9.3f}  "
              f"{np.median(p_arr):>10.2e}  "
              f"{np.median(a_arr):>10.2e}  "
              f"{np.median(i_arr):>10.0f}")


def _write_results_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        return
    fields = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    print(f"\nResults CSV: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    run_dir = _resolve_run_dir(LOG_DIR, EXPERIMENT_NAME, RUN_TIMESTAMP)
    ckpt_path = _resolve_checkpoint(run_dir, EPOCH)
    exp = _load_experiment(EXPERIMENT_NAME, EXPERIMENTS_REGISTRY,
                            fallback_dir=run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global MAX_ITER
    if MAX_ITER is None:
        k_stride = int(exp.get("k_stride_snapshots", 1))
        MAX_ITER = (TRAIN_N_STEPS + k_stride - 1) // k_stride + MAX_ITER_BUFFER
        print(f"[max_iter] auto: TRAIN_N_STEPS={TRAIN_N_STEPS}  "
              f"k_stride={k_stride}  → MAX_ITER={MAX_ITER}  "
              f"(+{MAX_ITER_BUFFER} buffer)")

    print(f"[experiment] {EXPERIMENT_NAME}: {exp.get('description', '')}")
    print(f"  run_dir   : {run_dir}")
    print(f"  ckpt      : {ckpt_path}  (EMA={USE_EMA_WEIGHTS})")

    if TARGET_OVERRIDE is not None:
        eval_rows = _files_from_target_override(Path(TARGET_OVERRIDE).resolve())
        print(f"  target    : OVERRIDE {TARGET_OVERRIDE}  "
              f"({len(eval_rows)} npz)")
    else:
        eval_rows = _load_eval_rows(exp["eval_manifest"], SYSTEMS_FILTER)
        print(f"  eval mfst : {exp['eval_manifest']}  "
              f"({len(eval_rows)} npz)")
    if SYSTEMS_FILTER:
        print(f"  systems   : {SYSTEMS_FILTER}")

    if not eval_rows:
        sys.exit("[abort] no eval files matched.")

    eval_root = run_dir / "eval"
    plot_dir = (Path(PLOT_DIR).resolve() if PLOT_DIR
                else (eval_root / "plots")) if SAVE_PLOTS else None
    xyz_dir = (Path(XYZ_DIR).resolve() if XYZ_DIR
                else (eval_root / "xyz")) if SAVE_XYZ else None
    csv_path = eval_root / "results.csv"
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
    if xyz_dir is not None:
        xyz_dir.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"  plots     : {plot_dir if SAVE_PLOTS else '(skipped)'}")
    print(f"  xyz       : {xyz_dir if SAVE_XYZ else '(skipped)'}")
    print(f"  results   : {csv_path}")
    print(f"  max_iter={MAX_ITER}  tol={CONVERGENCE_TOL_ANG} Å\n")

    model = _load_model(ckpt_path, device)

    results = []
    for r in eval_rows:
        try:
            results.append(evaluate_one(
                model, r["_npz_path"], device,
                plot_dir=plot_dir, xyz_dir=xyz_dir,
            ))
        except Exception as e:
            print(f"  {r['_npz_path'].name}: FAILED: {type(e).__name__}: {e}")

    _summarize(results)
    _write_results_csv(results, csv_path)


if __name__ == "__main__":
    main()
