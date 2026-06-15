"""Generate amorphous structures using the trained student model as the relaxer.

Same overall flow as scripts/macerelax/generation/generate_mace_trajectories.py
but replaces MACE+wall FIRE with iterative inference of the macerelax student
model trained by train_perl_ddp.py.

Per (CIF, regime, seed) trajectory:

  1. ase_read the CIF, build a tricor.Supercell at the regime's preset density.
  2. cell.generate(...) populates atoms.
  3. cell.bond_relax(...) cleanup removes overlaps before the model sees it.
  4. extract_shell_target_arrays(reference) gives the per-graph conditioning.
  5. Build a weight_vector for the regime (grain stats + density + wall_min +
     a placeholder fmax_initial — see DEFAULT_FMAX_INITIAL).
  6. Iterate the student model on the structure (up to MAX_ITER passes, with
     convergence check at CONVERGENCE_TOL_ANG).
  7. Save final positions + metadata as NPZ; optionally also CIF + XYZ.

Output per CIF lands in OUTPUT_ROOT/<compound>_<mp_id>_generated/.

USAGE WARNING (read MACE_CORPUS_FILTERING.md before generating on new chemistry)
The student was trained on a restricted element set.  Applying it to chemistries
outside that set may produce nonsensical relaxations.  Limit CIF_DIR /
CIF_LIST_FILE to the training chemistry (or chemistries you've validated
generalize via evaluate_perl.py).  No automatic safety check here.

Run — single GPU (small smoke test):
    /global/common/software/m5020/ehrdt/tricor/bin/python \\
        scripts/macerelax/generate_with_student.py

Run — multi-GPU (recommended for the full corpus).  Each worker handles a
round-robin slice of the CIF list, so 4 GPUs ≈ 4× throughput:

    # via torchrun (1 task, 4 workers):
    salloc -A m5241 -C "gpu&hbm80g" -q interactive -t 4:00:00 \\
        --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=none \\
        --cpus-per-task=64
    srun -l torchrun --nnodes=1 --nproc-per-node=4 \\
        /global/u2/e/ehrdt/tricor/scripts/macerelax/generate_with_student.py

    # OR via srun --ntasks (4 tasks, 1 GPU each):
    salloc -A m5241 -C "gpu&hbm80g" -q interactive -t 4:00:00 \\
        --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \\
        --cpus-per-task=16
    srun -l /global/common/software/m5020/ehrdt/tricor/bin/python \\
        /global/u2/e/ehrdt/tricor/scripts/macerelax/generate_with_student.py

Both detect the rank automatically.  Each worker writes to OUTPUT_ROOT
(no conflicts since each rank's CIF slice is disjoint).  The skip-if-
already-done logic still applies per rank, so a killed multi-GPU run
resumes cleanly when restarted.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# --- resource caps ---
# GPU_ID is only used as a fallback for single-process runs.  For multi-GPU
# parallelism, launch the script via torchrun or srun --ntasks=N — each
# worker picks up its own GPU from LOCAL_RANK / SLURM_LOCALID and processes
# a round-robin slice of the CIF list.  See "Multi-GPU launch" in the
# docstring at the top.
GPU_ID      = 0
NUM_THREADS = 4

# --- input ---
# Recommend: keep this constrained to training-chemistry CIFs.
CIF_DIR       = Path("/pscratch/sd/e/ehrdt/tricor/cifs_mp_cnos_le100meV_training")
# One filename per line.  None = use every *.cif in CIF_DIR.
CIF_LIST_FILE: Path | None = None
# Stop after this many CIFs (smoke-testing); None = process all.
MAX_CIFS: int | None = None

# --- model ---
MODEL_LOG_DIR        = "/pscratch/sd/e/ehrdt/macerelax/lightning_logs"
MODEL_RUN_NAME       = "ddp_v1"
MODEL_RUN_TIMESTAMP: str | None = None   # None → most recent run_*
MODEL_EPOCH          = "best"            # "last" | "best" | "<path>"
USE_EMA_WEIGHTS      = True

# ─── MODEL ARCHITECTURE — must match train_perl_ddp.py at training time ─────
MAX_Z                    = 120
NODE_DIM                 = 128
EDGE_DIM                 = 128
NUM_CONVS                = 4
WEIGHT_ENCODER_HIDDEN    = 64
SPECIES_PAIR_DIM         = 16
SHELL_TARGET_SPECIES_DIM = 8
SHELL_TARGET_HIDDEN      = 64
SHELL_TARGET_DROPOUT     = 0.0

# --- generation parameters (mirrors generate_mace_trajectories.py defaults) ---
REGIMES = ["amorphous", "SRO", "MRO", "LRO", "nanocrystalline", "crystalline_30"]
# One seed per regime, per CIF.  Use multiple entries for ensemble generation.
SEEDS = [2_000_000]
# Supercell dimensions in Å.  Cubic (a, a, a) at training time was 50.0;
# tuple of three lets you go non-cubic (e.g., (100., 100., 400.) for slabs).
# WARNING: large cells are OOD vs training; validate first via test scripts.
CELL_DIMS = (50.0, 50.0, 50.0)
DENSITY_BY_REGIME = {
    "amorphous":       0.92,
    "SRO":             0.92,
    "MRO":             0.88,
    "LRO":             0.92,
    "nanocrystalline": 0.96,
    "crystalline_30":  0.98,
}

# --- cleanup (bond_relax) ---
# Bond_relax is CPU-bound and dominates the per-CIF cost at large cells.
# 80 was the production default for training-data generation; 20 is enough
# to clear the worst initial overlaps and the student model handles the
# fine-grained relaxation.  ~4× speedup on this stage.
BOND_RELAX_N_ITER   = 20
BOND_RELAX_MAX_STEP = 0.1

# --- weight_vector construction ---
# The student saw fmax_initial as a per-trajectory conditioning input during
# training.  At generation time we don't run MACE so we don't have its true
# value; this default approximates the typical post-packing value (~5 eV/Å).
# Picking a value in the training distribution avoids OOD conditioning.
DEFAULT_FMAX_INITIAL = 5.0
WALL_MARGIN          = 0.0   # passed to per_pair_min_from_atoms

# --- inference ---
# MAX_ITER × k_stride should match the student's training horizon (k=5, N=60 → 12).
# Bumped to 15 to give the under-trained model a little extra cumulative
# displacement — but note this means iters 13-15 are slightly OOD vs training.
MAX_ITER             = 15
CONVERGENCE_TOL_ANG  = 0.001
CUTOFF               = 5.0

# --- output ---
OUTPUT_ROOT = Path("/pscratch/sd/e/ehrdt/macerelax/generated_v1")
SAVE_NPZ    = False  # turn back on if you want to retrain on these structures
SAVE_CIF    = False  # turn back on if you want per-traj CIFs
SAVE_XYZ    = True   # one-frame XYZ per trajectory (final structure only)
# Only relevant if XYZ_FINAL_ONLY is False below.
XYZ_ITER_STRIDE = 1
# True → write only the final frame to .xyz (compact, one-frame-per-traj).
# False → write initial + intermediates (stride XYZ_ITER_STRIDE) + final.
XYZ_FINAL_ONLY = True

# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Multi-GPU rank detection ────────────────────────────────────────────────
# When launched via torchrun (LOCAL_RANK/RANK/WORLD_SIZE set) or srun --ntasks=N
# (SLURM_LOCALID/SLURM_PROCID/SLURM_NTASKS set), each worker pins to its own
# GPU + processes a round-robin slice of the CIF list.  Single-process runs
# fall back to GPU_ID + processing all CIFs.
def _detect_rank_from_env() -> tuple[int, int, int]:
    """Return (local_rank, global_rank, world_size) — torchrun first, then SLURM, then 0/0/1."""
    if "LOCAL_RANK" in os.environ:
        return (int(os.environ["LOCAL_RANK"]),
                int(os.environ.get("RANK", os.environ["LOCAL_RANK"])),
                int(os.environ.get("WORLD_SIZE", 1)))
    if "SLURM_LOCALID" in os.environ:
        return (int(os.environ["SLURM_LOCALID"]),
                int(os.environ.get("SLURM_PROCID", os.environ["SLURM_LOCALID"])),
                int(os.environ.get("SLURM_NTASKS", 1)))
    return (0, 0, 1)


_LOCAL_RANK, _GLOBAL_RANK, _WORLD_SIZE = _detect_rank_from_env()
_MULTI_GPU = _WORLD_SIZE > 1

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# Single-process: mask only GPU_ID (the legacy behavior).  Multi-process:
# each worker sees all GPUs and picks LOCAL_RANK via torch.cuda.set_device().
if not _MULTI_GPU:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_ID))

# Allocator config — expandable_segments reduces fragmentation from the
# variable-shape graphs across CIFs (different atom counts each).
# garbage_collection_threshold:0.8 proactively releases unused fragments
# at 80% memory.  Both are needed at 100×100×400 scale where the student
# forward can request 5-10 GB tensors in flight.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)
_n = str(NUM_THREADS)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _n)

import csv
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
torch.set_num_threads(NUM_THREADS)
from torch_geometric.data import Batch

from ase.io import read as ase_read, write as ase_write

import tricor as tc
from tricor.flowmatch.flow_utils import (
    periodic_radius_graph_cell_list,
    periodic_radius_graph_chunked,
)
from tricor.macerelax.model import RelaxMLModel
from tricor.macerelax.data import (
    ShellTargetData,
    _weight_vector_from_row,
)
from tricor.shells import CoordinationShellTarget
from tricor.relaxml.shell_target import extract_shell_target_arrays

# wall_calculator's per_pair_min_from_atoms gives us the wall_global_min for
# the weight_vector — same routine the production generation uses.
_GEN_DIR = Path(__file__).resolve().parent / "generation"
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))
from wall_calculator import per_pair_min_from_atoms


# ─────────────────────────────────────────────────────────────────────────────
# Tricor preset overrides (matches generate_mace_trajectories.py)
# ─────────────────────────────────────────────────────────────────────────────

_MP_ID_RE = re.compile(r"^(mp-\d+)_(.+)$")


def _build_local_presets() -> dict:
    """tricor PRESETS with displacement_sigma=0 + a custom crystalline_30."""
    base: dict = {}
    for name in REGIMES:
        if name == "crystalline_30":
            base[name] = dict(
                num_steps=0, grain_size=30.0, displacement_sigma=0.0,
                bond_weight=3.0, angle_weight=1.5,
            )
        else:
            d = dict(tc.Supercell.PRESETS[name])
            d["displacement_sigma"] = 0.0
            base[name] = d
    return base


LOCAL_PRESETS = _build_local_presets()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (matches evaluate_perl.py)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_run_dir(log_dir: str, run_name: str,
                     run_timestamp: str | None) -> Path:
    exp_dir = Path(log_dir) / run_name
    if not exp_dir.is_dir():
        raise SystemExit(f"[abort] no training logs at {exp_dir}")
    if run_timestamp is not None:
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
    if epoch not in ("last", "best") and Path(epoch).is_file():
        return Path(epoch).resolve()
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"[abort] no checkpoints/ under {run_dir}")
    target = ckpt_dir / f"{epoch}.pt"
    if not target.is_file():
        alt = "last" if epoch == "best" else "best"
        alt_path = ckpt_dir / f"{alt}.pt"
        if alt_path.is_file():
            print(f"[warn] {target.name} not found, falling back to {alt}.pt")
            return alt_path.resolve()
        raise SystemExit(f"[abort] no {epoch}.pt (or alternative) in {ckpt_dir}")
    return target.resolve()


def _build_model() -> RelaxMLModel:
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
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = _build_model()
    for p in model.processor.edge_norms[-1].parameters():
        p.requires_grad_(False)
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


# ─────────────────────────────────────────────────────────────────────────────
# Supercell construction (mirrors compare_mace_teachers.py / production)
# ─────────────────────────────────────────────────────────────────────────────

def parse_cif_name(cif_path: Path) -> tuple[str, str]:
    m = _MP_ID_RE.match(cif_path.stem)
    if not m:
        return "", cif_path.stem
    return m.group(1), m.group(2)


def build_supercell(cif_path: Path, regime: str, rho: float, seed: int):
    """Pack a tricor supercell at the requested regime/density."""
    ref = ase_read(str(cif_path), format="cif")
    shell = tc.CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)

    from tricor import G3Distribution
    dist = G3Distribution(ref, label=str(cif_path.stem))
    dist.measure_g3(r_max=10.0, r_step=0.1, phi_num_bins=90, show_progress=False)

    cell = tc.Supercell(
        dist,
        cell_dim_angstroms=tuple(CELL_DIMS),
        relative_density=rho,
        rng_seed=seed,
        label=f"{cif_path.stem}_{regime}_{seed}",
    )
    preset = dict(LOCAL_PRESETS[regime])
    preset["num_steps"] = 0  # skip shell_relax — bond_relax cleanup follows
    summary = cell.generate(
        shell, **preset, refine_orientations=False, show_progress=False,
    )
    return cell, shell, ref, summary


def cleanup(cell, shell):
    cell.bond_relax(shell, n_iter=BOND_RELAX_N_ITER, max_step=BOND_RELAX_MAX_STEP)


# ─────────────────────────────────────────────────────────────────────────────
# Weight vector + shell_target (per-trajectory conditioning)
# ─────────────────────────────────────────────────────────────────────────────

def build_weight_vector(cell, summary: dict, regime: str, rho: float) -> np.ndarray:
    """Mimic the conditioning fields that NPZ rows store at training time."""
    grain_size = float(summary.get("grain_size") or 0.0)
    num_grains = int(summary.get("n_grains") or 0)
    crystalline_fraction = float(summary.get("crystalline_fraction") or 0.0)

    # wall_global_min is the smallest per-pair wall threshold derived from
    # the cleaned initial structure.  Use the same routine as production.
    r_min_per_pair = per_pair_min_from_atoms(cell.atoms, margin=WALL_MARGIN)
    wall_global_min = (float(min(r_min_per_pair.values()))
                        if r_min_per_pair else 0.0)

    fake_row = {
        "grain_size":           str(grain_size),
        "num_grains":           str(num_grains),
        "crystalline_fraction": str(crystalline_fraction),
        "rel_density":          str(rho),
        "wall_global_min":      str(wall_global_min),
        "fmax_initial":         str(DEFAULT_FMAX_INITIAL),
    }
    return _weight_vector_from_row(fake_row)


def build_shell_target(ref) -> dict:
    """Compute the four shell_target arrays from the reference structure."""
    target = CoordinationShellTarget.from_atoms(ref)
    return extract_shell_target_arrays(target)


# ─────────────────────────────────────────────────────────────────────────────
# Iterative student-model inference
# ─────────────────────────────────────────────────────────────────────────────

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


def _wrap_positions(pos, cell):
    inv_cell = torch.linalg.inv(cell)
    frac = pos @ inv_cell.T
    frac = frac - torch.floor(frac)
    return frac @ cell


@torch.no_grad()
def run_iterative_inference(
    model, initial_positions, cell, species, weight_vector, shell_target,
    device, *, cutoff=CUTOFF, max_iter=MAX_ITER, tol=CONVERGENCE_TOL_ANG,
    collect_every=0,
):
    """Iteratively apply the student model until convergence or max_iter.

    Returns (final_positions, n_iter_run, intermediates).
    """
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
            f"    iter {it:2d}: |delta| mean={d_norms.mean().item():.4f}  "
            f"max={d_norms.max().item():.4f}  Å"
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


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenResult:
    cif_filename:  str
    compound:      str
    mp_id:         str
    regime:        str
    rng_seed:      int
    n_atoms:       int
    n_iter:        int
    runtime_sec:   float
    out_npz:       str
    error:         str = ""


def save_outputs(out_dir: Path, traj: GenResult, initial: np.ndarray,
                  final: np.ndarray, intermediates: list[tuple[int, np.ndarray]],
                  cell_arr: np.ndarray, species_numbers: np.ndarray,
                  weight_vector: np.ndarray, shell_target: dict,
                  regime: str, rho: float, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{traj.compound}_{traj.mp_id}_{regime}_seed{traj.rng_seed}"

    if SAVE_NPZ:
        npz_path = out_dir / f"{base}.npz"
        np.savez_compressed(
            npz_path,
            initial_positions=initial.astype(np.float32),
            final_positions=final.astype(np.float32),
            cell=cell_arr.astype(np.float32),
            species_numbers=species_numbers.astype(np.int32),
            weight_vector=weight_vector.astype(np.float32),
            shell_pair_species=shell_target["shell_pair_species"],
            shell_pair_features=shell_target["shell_pair_features"],
            shell_triplet_species=shell_target["shell_triplet_species"],
            shell_triplet_features=shell_target["shell_triplet_features"],
            compound=np.asarray(traj.compound),
            mp_id=np.asarray(traj.mp_id),
            regime=np.asarray(regime),
            rng_seed=np.int64(traj.rng_seed),
            n_iter=np.int32(traj.n_iter),
            rel_density=np.float32(rho),
            grain_size=np.float32(summary.get("grain_size") or 0.0),
            num_grains=np.int32(summary.get("n_grains") or 0),
            crystalline_fraction=np.float32(summary.get("crystalline_fraction") or 0.0),
            source=np.asarray("student_generated_v1"),
        )
        traj.out_npz = str(npz_path)

    if SAVE_CIF:
        from ase import Atoms
        atoms = Atoms(
            numbers=species_numbers,
            positions=final.astype(np.float64),
            cell=cell_arr.astype(np.float64),
            pbc=True,
        )
        ase_write(str(out_dir / f"{base}.cif"), atoms, format="cif")

    if SAVE_XYZ:
        from ase import Atoms
        if XYZ_FINAL_ONLY:
            # One-frame XYZ of the final relaxed structure.
            atoms = Atoms(
                numbers=species_numbers,
                positions=np.asarray(final, dtype=np.float64),
                cell=cell_arr.astype(np.float64),
                pbc=True,
            )
            atoms.info["frame_label"] = "final"
            atoms.info["regime"] = regime
            atoms.info["n_iter"] = int(traj.n_iter)
            ase_write(str(out_dir / f"{base}.xyz"), atoms, format="extxyz")
        else:
            # Multi-frame trajectory: initial + intermediates + final.
            frames = []
            for label, pos, it_idx in (
                ("initial", initial, None),
                *((f"iter_{it}", p, it) for it, p in intermediates),
                ("final", final, None),
            ):
                atoms = Atoms(
                    numbers=species_numbers,
                    positions=np.asarray(pos, dtype=np.float64),
                    cell=cell_arr.astype(np.float64),
                    pbc=True,
                )
                atoms.info["frame_label"] = label
                atoms.info["regime"] = regime
                if it_idx is not None:
                    atoms.info["iter"] = int(it_idx)
                frames.append(atoms)
            ase_write(str(out_dir / f"{base}.xyz"), frames, format="extxyz")


# ─────────────────────────────────────────────────────────────────────────────
# Per-CIF orchestration
# ─────────────────────────────────────────────────────────────────────────────

def generate_for_cif(model, cif_path: Path, device) -> list[GenResult]:
    """Generate one trajectory per (regime, seed) for this CIF."""
    mp_id, compound = parse_cif_name(cif_path)
    sys_label = f"{compound}_{mp_id}" if mp_id else compound
    out_dir = OUTPUT_ROOT / f"{sys_label}_generated"

    # Build the per-CIF reference once: shell_target arrays are constant
    # across regimes/seeds for a given CIF.
    try:
        ref = ase_read(str(cif_path), format="cif")
        shell_arrays = build_shell_target(ref)
    except Exception as exc:
        print(f"[{sys_label}] FAIL reading CIF / building shell_target: {exc}")
        traceback.print_exc(limit=3, file=sys.stdout)
        return []

    results: list[GenResult] = []
    for regime in REGIMES:
        rho = DENSITY_BY_REGIME.get(regime)
        if rho is None:
            print(f"[{sys_label}] no rel_density for regime {regime!r}; skipping")
            continue
        for seed in SEEDS:
            # ── Resume / skip-if-already-done ───────────────────────────
            # If the output file already exists from a previous run, skip
            # this trajectory.  Use the same naming convention as save_outputs.
            base = f"{compound}_{mp_id}_{regime}_seed{seed}"
            existing = []
            if SAVE_XYZ:
                existing.append(out_dir / f"{base}.xyz")
            if SAVE_NPZ:
                existing.append(out_dir / f"{base}.npz")
            if SAVE_CIF:
                existing.append(out_dir / f"{base}.cif")
            if existing and all(p.is_file() for p in existing):
                print(f"  [{sys_label}/{regime}/seed{seed}] skip (already done)",
                      flush=True)
                continue

            t0 = time.time()
            print(f"  [{sys_label}/{regime}/seed{seed}] running...", flush=True)
            try:
                # Stage timers — these are added to the success print so each
                # CIF tells us exactly where its wall-clock went.  Useful for
                # spotting whether tricor packing, bond_relax, or student
                # inference is the active bottleneck on any given system.
                _ts = time.time()
                cell, shell, ref_atoms, summary = build_supercell(
                    cif_path, regime, rho, seed,
                )
                t_pack = time.time() - _ts

                _ts = time.time()
                cleanup(cell, shell)
                t_cleanup = time.time() - _ts

                _ts = time.time()
                atoms = cell.atoms
                initial_pos = atoms.positions.copy()
                species_numbers = atoms.numbers.copy()
                cell_arr = atoms.cell.array.copy()

                weight_vector = build_weight_vector(cell, summary, regime, rho)
                t_setup = time.time() - _ts

                _ts = time.time()
                # Skip intermediate-frame collection if we won't write them.
                collect = XYZ_ITER_STRIDE if (SAVE_XYZ and not XYZ_FINAL_ONLY) else 0
                final_pos, n_iter, intermediates = run_iterative_inference(
                    model, initial_pos, cell_arr, species_numbers,
                    weight_vector, shell_arrays, device,
                    collect_every=collect,
                )
                t_model = time.time() - _ts

                traj = GenResult(
                    cif_filename=cif_path.name,
                    compound=compound,
                    mp_id=mp_id,
                    regime=regime,
                    rng_seed=seed,
                    n_atoms=int(len(species_numbers)),
                    n_iter=int(n_iter),
                    runtime_sec=float(time.time() - t0),
                    out_npz="",
                )
                _ts = time.time()
                save_outputs(
                    out_dir, traj, initial_pos, final_pos, intermediates,
                    cell_arr, species_numbers, weight_vector, shell_arrays,
                    regime, rho, summary,
                )
                t_save = time.time() - _ts
                results.append(traj)
                print(f"    ✓ atoms={traj.n_atoms}  iters={n_iter}/{MAX_ITER}  "
                      f"pack={t_pack:.1f}s  cleanup={t_cleanup:.1f}s  "
                      f"setup={t_setup:.1f}s  model={t_model:.1f}s  "
                      f"save={t_save:.1f}s  total={traj.runtime_sec:.1f}s")
            except torch.cuda.OutOfMemoryError as exc:
                # OOM on huge cells (e.g. dense chemistry at 100×100×400) —
                # skip this trajectory, clear the cache, keep going.  Other
                # regimes/seeds for this CIF may still fit (smaller graphs
                # for some regimes).
                torch.cuda.empty_cache()
                err = f"OutOfMemoryError"
                print(f"    ✗ OOM (skip — see PYTORCH_CUDA_ALLOC_CONF for tuning)",
                      flush=True)
                results.append(GenResult(
                    cif_filename=cif_path.name, compound=compound, mp_id=mp_id,
                    regime=regime, rng_seed=seed, n_atoms=-1, n_iter=0,
                    runtime_sec=time.time() - t0, out_npz="", error=err,
                ))
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                traceback.print_exc(limit=3, file=sys.stdout)
                results.append(GenResult(
                    cif_filename=cif_path.name, compound=compound, mp_id=mp_id,
                    regime=regime, rng_seed=seed, n_atoms=-1, n_iter=0,
                    runtime_sec=time.time() - t0, out_npz="", error=err,
                ))
                print(f"    ✗ {err}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def list_cifs() -> list[Path]:
    if not CIF_DIR.is_dir():
        raise SystemExit(f"[abort] CIF_DIR not found: {CIF_DIR}")
    if CIF_LIST_FILE is not None and Path(CIF_LIST_FILE).is_file():
        names = [
            line.strip() for line in Path(CIF_LIST_FILE).read_text().splitlines()
            if line.strip()
        ]
        paths = [CIF_DIR / n for n in names]
    else:
        paths = sorted(CIF_DIR.glob("*.cif"))
    paths = [p for p in paths if p.is_file()]
    if MAX_CIFS is not None:
        paths = paths[:MAX_CIFS]
    return paths


def main() -> None:
    # Only rank 0 creates the root + prints headers — avoids racy mkdir
    # and duplicate output from N workers.
    if _GLOBAL_RANK == 0:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Pin this process's CUDA device under multi-GPU launches.
    if _MULTI_GPU and torch.cuda.is_available():
        torch.cuda.set_device(_LOCAL_RANK)
        device = torch.device("cuda", _LOCAL_RANK)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    run_dir = _resolve_run_dir(MODEL_LOG_DIR, MODEL_RUN_NAME, MODEL_RUN_TIMESTAMP)
    ckpt_path = _resolve_checkpoint(run_dir, MODEL_EPOCH)

    if _GLOBAL_RANK == 0:
        print(f"[model] run_dir = {run_dir}")
        print(f"[model] ckpt    = {ckpt_path}  (EMA={USE_EMA_WEIGHTS})")
        print(f"[model] arch    = "
              f"node={NODE_DIM} edge={EDGE_DIM} convs={NUM_CONVS}")
        if _MULTI_GPU:
            print(f"[parallel] world_size={_WORLD_SIZE} "
                  f"(round-robin CIF partitioning)")
        else:
            print(f"[parallel] single-process")
    print(f"[rank {_GLOBAL_RANK}/{_WORLD_SIZE}] device={device}", flush=True)

    cif_paths_all = list_cifs()
    if not cif_paths_all:
        raise SystemExit(f"[abort] no CIFs to process under {CIF_DIR}")

    # Round-robin partition across workers — each rank handles cif_idx % N == rank.
    # Better than chunk-partition for uneven workloads (some CIFs are bigger).
    cif_paths = cif_paths_all[_GLOBAL_RANK::_WORLD_SIZE]

    n_per_cif = len(REGIMES) * len(SEEDS)
    if _GLOBAL_RANK == 0:
        print(f"[corpus] {len(cif_paths_all)} CIFs total × {len(REGIMES)} regimes "
              f"× {len(SEEDS)} seeds = {len(cif_paths_all) * n_per_cif} trajectories")
        print(f"[output] root   = {OUTPUT_ROOT}")
    print(f"[rank {_GLOBAL_RANK}/{_WORLD_SIZE}] my slice: "
          f"{len(cif_paths)} CIFs × {n_per_cif} = "
          f"{len(cif_paths) * n_per_cif} trajectories", flush=True)
    print()

    model = _load_model(ckpt_path, device)

    all_results: list[GenResult] = []
    t_start = time.time()
    for i, cif_path in enumerate(cif_paths, 1):
        print(f"[{i}/{len(cif_paths)}] {cif_path.name}")
        rs = generate_for_cif(model, cif_path, device)
        all_results.extend(rs)
        elapsed = time.time() - t_start
        n_done = i * n_per_cif   # nominal target
        rate = i / max(elapsed, 1e-9) * 60.0
        eta_sec = (len(cif_paths) - i) * (elapsed / max(i, 1))
        print(f"  cumulative: {len(all_results)} trajs done  "
              f"rate={rate:.1f} CIF/min  ETA={int(eta_sec/60)} min")
        print()

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUTPUT_ROOT / "summary.csv"
    if all_results:
        with open(summary_csv, "w", newline="") as f:
            fields = list(asdict(all_results[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        print(f"[summary] wrote {summary_csv}")

    # ── Final breakdown ──────────────────────────────────────────────────
    n_ok = sum(1 for r in all_results if not r.error)
    n_fail = len(all_results) - n_ok
    print()
    print("=" * 60)
    print(f"  Total trajectories : {len(all_results)}")
    print(f"  Succeeded          : {n_ok}")
    print(f"  Failed             : {n_fail}")
    print(f"  Total wall-clock   : {(time.time() - t_start) / 60:.1f} min")
    print(f"  Output root        : {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
