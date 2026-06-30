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

# ── NERSC CNOS PRODUCTION preset (from the macerelax/NERSC line, merged 2026-06-24) ──
# Run-specific values used for the A100 CNOS production run. To use, set these
# in the matching sections below (they differ from the PERLMUTTER defaults):
#   CIF_DIR        = Path("/pscratch/sd/e/ehrdt/all_cnos_cifs")
#   CIF_LIST_FILE  = Path("/global/u2/e/ehrdt/tricor/scratch/incomplete_cifs.txt")
#   MAX_ITER             = 12     # exactly the k=5, N=60 training horizon (no OOD iters 13-15)
#   USE_BF16_INFERENCE   = True   # A100 bf16 ≈ 16× fp32 — worth it here (was a loss on buffle/Blackwell)
#   OUTPUT_ROOT    = Path("/pscratch/sd/e/ehrdt/macerelax/generated_cnos_v1")

# --- input ---
# Recommend: keep this constrained to training-chemistry CIFs.
# ── BUFFLE E2E VALIDATION (commented out — uncomment for local buffle tests) ─
# CIF_DIR       = Path("/home/ehrdt/cifs_mp_exp_le100meV")
# CIF_LIST_FILE: Path | None = Path("/home/ehrdt/tricor/scratch/validate_e2e_cifs.txt")
# ── PERLMUTTER PRODUCTION ───────────────────────────────────────────────────
CIF_DIR       = Path("/pscratch/sd/e/ehrdt/tricor/cifs_mp_cnos_le100meV_training")
# One filename per line.  None = use every *.cif in CIF_DIR.
CIF_LIST_FILE: Path | None = None
# Stop after this many CIFs (smoke-testing); None = process all.
MAX_CIFS: int | None = None

# --- model ---
# ── BUFFLE E2E VALIDATION (commented out — uncomment for local buffle tests) ─
# MODEL_LOG_DIR        = "/home/ehrdt"
# MODEL_RUN_NAME       = "perl_tb_logs"
# MODEL_RUN_TIMESTAMP: str | None = "1781193088"  # known-good checkpoint, Jun 11
# ── PERLMUTTER PRODUCTION ───────────────────────────────────────────────────
MODEL_LOG_DIR        = "/pscratch/sd/e/ehrdt/macerelax/lightning_logs"
MODEL_RUN_NAME       = "ddp_v1"
MODEL_RUN_TIMESTAMP: str | None = None   # None → most recent run_*
MODEL_EPOCH          = "best"            # "last" | "best" | "<path>"
USE_EMA_WEIGHTS      = True
# Fail loudly if the checkpoint's weights don't line up with the model
# architecture built from the constants below.  Because load_state_dict
# runs with strict=False, a mismatch (e.g. the MODEL ARCHITECTURE knobs
# drifting from train_perl_ddp.py) would otherwise be silently tolerated
# and generate garbage from partially-random weights.  Set False only if
# you have a KNOWN-benign missing/unexpected key set.
STRICT_CHECKPOINT_LOAD = True

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
# Seeds drawn per regime per CIF.  Add entries to increase ensemble size.
SEEDS = [2_000_000, 2_000_001]   # 2 structures per regime per CIF
# Supercell dimensions in Å.  Cubic (a, a, a) at training time was 50.0;
# tuple of three lets you go non-cubic (e.g., (100., 100., 400.) for slabs).
# WARNING: large cells are OOD vs training; validate first via test scripts.
CELL_DIMS = (100.0, 100.0, 400.0)
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
# Edge-chunk size for the MeshGraphNetsConv ``EdgeProcessor``.  The
# (E, 384) cat tensor inside each conv layer's edge processor would
# otherwise peak at ~46 GB for Fe2N at 100×100×400 (E≈30 M edges) and
# OOM on A100 80 GB.  Chunking processes the per-edge MLP in batches
# of ``EDGE_CHUNK_SIZE`` edges, writing into a pre-allocated output
# tensor in place.  Output agrees with the unchunked path to ~1e-4 Å
# max (FP-summation-order noise inside the LayerNorm + matmuls), well
# below our g(r) noise floor.  Measured peaks (Fe2N amorphous,
# 100×100×400, 30 M edges):
#   unchunked      : 93 GB   ← OOM on A100 80 GB
#   chunk = 8 M    : 87 GB   ← borderline
#   chunk = 4 M    : 79 GB   ← under A100 limit
#   chunk = 2 M    : 76 GB   ← safe margin (recommended default)
#   chunk = 1 M    : 74 GB   ← floor; further chunking doesn't help
# Smaller chunks add ~no wall-clock overhead (sequential MLP calls
# inside one CUDA stream).  Set to 0 to disable chunking entirely.
EDGE_CHUNK_SIZE      = 2_000_000
# ``torch.compile`` over the student model.  Default off because
# empirical benchmarking on Fe2N (30 M edges) showed compile mode
# ``reduce-overhead`` + ``dynamic=True`` ran SLOWER than eager
# (20 s → 27 s) — the chunked EdgeProcessor loop confuses the
# tracer + dynamic-shape recompiles + CUDA-graph overhead outweigh
# kernel fusion at this scale.  Left as a CONFIG knob in case future
# PyTorch versions handle it better, or if you want to experiment
# with ``mode="default"``.
USE_TORCH_COMPILE    = False
TORCH_COMPILE_MODE   = "reduce-overhead"   # or "default" / "max-autotune"
# Convert model + inputs to bfloat16 to exploit A100 Tensor Cores.
# Theory: A100 has 312 TFLOPS bf16 vs 19.5 TFLOPS fp32 (16× gap on
# matmuls).  Buffle's Blackwell has only ~2× gap, so bf16 was a loss
# there — but on A100 the speedup typically outweighs the dtype
# bookkeeping.  Tradeoff: position updates accumulate ~few mÅ of bf16
# rounding noise per iter over the 15-iter loop.  Validate via a
# fp32-vs-bf16 comparison run on one CIF before flipping to True
# corpus-wide.  Position tensor (``pos``) stays in fp32 throughout so
# the running update doesn't compound bf16 rounding into the position
# array itself.
USE_BF16_INFERENCE   = False

# --- output ---
# ── BUFFLE E2E VALIDATION (commented out — uncomment for local buffle tests) ─
# OUTPUT_ROOT = Path("/home/ehrdt/tricor/scratch/validate_e2e_out")
# ── PERLMUTTER PRODUCTION ───────────────────────────────────────────────────
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

import atexit
import csv
import json
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict, fields as dc_fields
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import torch
torch.set_num_threads(NUM_THREADS)
from torch_geometric.data import Batch

from ase.io import read as ase_read, write as ase_write

import tricor as tc
from tricor.macerelax.flow_utils import (
    periodic_radius_graph_cell_list,
    periodic_radius_graph_chunked,
)
from tricor.macerelax.model import RelaxMLModel
from tricor.macerelax.data import (
    ShellTargetData,
    _weight_vector_from_row,
)
from tricor.shells import CoordinationShellTarget
from tricor.macerelax.shell_target import extract_shell_target_arrays

# wall_calculator's per_pair_min_from_atoms gives us the wall_global_min for
# the weight_vector — same routine the production generation uses.
_GEN_DIR = Path(__file__).resolve().parent / "generation"
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))
from wall_calculator import per_pair_min_from_atoms


# ─────────────────────────────────────────────────────────────────────────────
# Run-manifest + provenance helpers
# ─────────────────────────────────────────────────────────────────────────────
#
# Every generation run writes a single ``_runs/{run_id}.json`` capturing the
# full CONFIG snapshot, git state, model checkpoint info, and SLURM env.  Each
# trajectory carries the same ``run_id`` so per-traj rows in ``manifest.csv``
# / NPZ / XYZ can be joined back to "exactly which parameters produced this
# structure?" without opening the file.
#
# Layout under OUTPUT_ROOT:
#
#   _runs/{run_id}.json              ← config + git + slurm + stats
#   _runs/{run_id}.rank{N}.csv       ← per-rank summary; rank 0 reads all at end
#   <compound>_<mp-id>_generated/
#       manifest.csv                 ← per-CIF, appended as trajectories finish
#       *.npz / *.xyz                ← each carries run_id

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def _git_is_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def derive_run_id() -> str:
    """Stable identifier shared by every rank of the same launch.

    Uses SLURM_JOB_ID under SLURM (identical across ranks).  Falls back to
    a UTC hour timestamp + PID for local runs (single-rank only — multi-rank
    local launches without SLURM will disagree on PID, which is fine for
    dev/test).  Always suffixed by the short git SHA so a run started from
    a different commit is distinguishable from one started here.
    """
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        prefix = f"slurm{slurm}"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        prefix = f"{ts}_pid{os.getpid()}"
    return f"{prefix}_{_git_sha_short()}"


def collect_config_snapshot() -> dict:
    """Capture every module-level UPPER_CASE constant in a JSON-safe dict."""
    g = globals()
    snap: dict = {}
    for name, value in g.items():
        if not name or not name[0].isupper() or name.startswith("_"):
            continue
        if isinstance(value, Path):
            snap[name] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            snap[name] = value
        elif isinstance(value, (list, tuple)):
            snap[name] = list(value)
        elif isinstance(value, dict):
            snap[name] = {str(k): v for k, v in value.items()}
        # anything else (modules, callables) silently skipped
    return snap


def write_run_manifest_start(run_id: str, ckpt_path: Path) -> Path:
    """Write the initial ``_runs/{run_id}.json`` with status=running.

    Rank-0 only — every rank computes the same ``run_id``, but only rank 0
    materializes the manifest to avoid four ranks racing on the same file.
    """
    manifest_path = OUTPUT_ROOT / "_runs" / f"{run_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        st = ckpt_path.stat()
        ckpt_size = int(st.st_size)
        ckpt_mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        ckpt_size = -1
        ckpt_mtime = ""
    payload = {
        "run_id":          run_id,
        "schema_version":  1,
        "status":          "running",
        "started_at_utc":  datetime.now(timezone.utc).isoformat(),
        "ended_at_utc":    None,
        "git": {
            "sha":   _git_sha_short(),
            "dirty": _git_is_dirty(),
        },
        "slurm": {
            "job_id":    os.environ.get("SLURM_JOB_ID"),
            "nnodes":    int(os.environ.get("SLURM_NNODES", 1)),
            "ntasks":    int(os.environ.get("SLURM_NTASKS", 1)),
            "node_list": os.environ.get("SLURM_NODELIST", ""),
        },
        "world_size":      _WORLD_SIZE,
        "model": {
            "log_dir":              MODEL_LOG_DIR,
            "run_name":             MODEL_RUN_NAME,
            "run_timestamp":        MODEL_RUN_TIMESTAMP,
            "epoch":                MODEL_EPOCH,
            "checkpoint_path":      str(ckpt_path),
            "checkpoint_size_bytes": ckpt_size,
            "checkpoint_mtime_utc":  ckpt_mtime,
            "use_ema_weights":      USE_EMA_WEIGHTS,
        },
        "config":          collect_config_snapshot(),
        "stats":           {},
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str))
    return manifest_path


def update_run_manifest_end(manifest_path: Path, *, status: str,
                            stats: dict) -> None:
    """Patch the manifest at end of run with final stats + status."""
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        return
    payload["status"] = status
    payload["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload.setdefault("stats", {}).update(stats)
    manifest_path.write_text(json.dumps(payload, indent=2, default=str))


def _install_interrupt_handlers(manifest_path: Path) -> None:
    """Install atexit + SIGTERM/SIGINT handlers that flip status if running.

    Covers the *graceful* death cases: Ctrl+C (SIGINT), SLURM time-limit
    warning (SIGTERM ~ 30 s before SIGKILL), Python uncaught exception
    bubbling out of main().  SIGKILL / OOM-killer / node crash cannot be
    intercepted by any handler — those get cleaned up by
    ``reconcile_stale_running_manifests`` on the next launch.
    """
    def _flush_on_exit() -> None:
        try:
            if not manifest_path.is_file():
                return
            payload = json.loads(manifest_path.read_text())
            if payload.get("status") != "running":
                return  # already completed/interrupted/etc — don't clobber
            payload["status"] = "interrupted"
            payload["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            payload.setdefault("stats", {})["interrupted_on_exit"] = True
            manifest_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            pass

    atexit.register(_flush_on_exit)

    def _signal_handler(signum, _frame):
        _flush_on_exit()
        # Restore the default handler and re-raise so the process actually
        # dies the way the user requested (otherwise we'd silently swallow
        # the signal).
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # Can't install — e.g., not running on the main thread.  The
            # atexit hook still covers normal-exit paths.
            pass


def _slurm_job_alive(job_id: str | None) -> Optional[bool]:
    """Return True / False / None — None means we genuinely can't tell.

    True  : squeue reports the job as queued/running.
    False : sacct shows the job has ended (FAILED / CANCELLED / TIMEOUT / …).
    None  : SLURM not available, neither tool returns info, or query failed.
    """
    if not job_id:
        return None
    try:
        out = subprocess.check_output(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return None
    if out:
        return True
    # squeue empty — confirm via sacct that the job actually existed and
    # terminated (vs simply never being known to slurm).
    try:
        out = subprocess.check_output(
            ["sacct", "-j", str(job_id), "-X", "-n", "-o", "State"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    return False


def reconcile_stale_running_manifests(*, max_age_hours: float = 48.0) -> dict:
    """Scan ``_runs/*.json`` and flip visibly-dead 'running' entries.

    Detection rules (per-manifest):
      * SLURM job ID present  → query squeue/sacct.  If alive: leave alone.
        If clearly terminated: mark ``stale_running``.  If unknown: fall
        through to the age check.
      * No SLURM job ID (or SLURM unqueryable)  → use ``started_at_utc``.
        Older than ``max_age_hours`` → mark ``stale_running``.

    The conservative path is "leave alone" — false positives would clobber
    a real running job's manifest, which is much worse than leaving a stale
    entry around for one more run cycle.

    Returns ``{scanned, marked, left}`` for the caller's log line.
    """
    runs_dir = OUTPUT_ROOT / "_runs"
    if not runs_dir.is_dir():
        return {"scanned": 0, "marked": 0, "left": 0}

    now = datetime.now(timezone.utc)
    max_age = timedelta(hours=max_age_hours)
    scanned = 0
    marked = 0
    left = 0

    for path in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get("status") != "running":
            continue
        scanned += 1
        job_id = (payload.get("slurm") or {}).get("job_id")

        is_stale = False
        reason = ""

        alive = _slurm_job_alive(job_id)
        if alive is True:
            left += 1
            continue
        if alive is False:
            is_stale = True
            reason = "slurm_job_terminated"

        if not is_stale:
            # Fall back to age threshold when SLURM doesn't know about it
            # (or there's no SLURM job id at all — local dev runs).
            started_str = payload.get("started_at_utc") or ""
            try:
                started = datetime.fromisoformat(started_str)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
            except Exception:
                started = None
            if started is not None and (now - started) > max_age:
                is_stale = True
                reason = f"older_than_{max_age_hours:.0f}h_no_slurm_info"

        if not is_stale:
            left += 1
            continue

        payload["status"] = "stale_running"
        payload["ended_at_utc"] = now.isoformat()
        payload.setdefault("stats", {})["stale_detected_at_next_run_start"] = True
        payload["stats"]["stale_reason"] = reason
        try:
            path.write_text(json.dumps(payload, indent=2, default=str))
            marked += 1
        except Exception:
            pass

    return {"scanned": scanned, "marked": marked, "left": left}


def _gen_result_fieldnames() -> list[str]:
    """Stable column order for any manifest derived from GenResult."""
    return [f.name for f in dc_fields(GenResult)]


def append_per_cif_manifest_row(out_dir: Path, row: "GenResult") -> None:
    """Append one trajectory row to ``out_dir/manifest.csv``.

    Called from inside ``generate_for_cif`` as each trajectory finishes
    (success / OOM / error), NOT once per CIF.  This guarantees a row
    exists on disk by the time the next trajectory starts — so a mid-CIF
    crash (OOM-killer, SLURM time-out, node failure) doesn't orphan the
    trajectories that DID complete:

        bulk-at-end (old)            row-per-traj (new)
        ──────────────────           ──────────────────
        trajs 0-5  ✓                 trajs 0-5  ✓ + rows on disk
        traj  6    OOM-killed        traj  6    OOM-killed
        manifest never written       manifest already has rows 0-5

    Resume-safe: creates the header on first call, appends otherwise.
    Safe under multi-rank launches because each rank's CIF slice is
    disjoint — no two ranks ever append to the same file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.csv"
    fieldnames = _gen_result_fieldnames()
    existed = path.is_file()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if not existed:
            w.writeheader()
        w.writerow(asdict(row))


def write_per_rank_summary(run_id: str, results: list["GenResult"]) -> Path:
    """Write the rank's own contribution to ``_runs/{run_id}.rank{N}.csv``.

    Replaces the previous race-prone shared ``summary.csv`` write.  The
    flat-table builder (``build_dataset_table.py``) joins all per-rank
    summaries plus the run JSON to produce the corpus-wide table.
    """
    path = OUTPUT_ROOT / "_runs" / f"{run_id}.rank{_GLOBAL_RANK}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _gen_result_fieldnames()
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    return path


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
        if STRICT_CHECKPOINT_LOAD:
            raise RuntimeError(
                f"Checkpoint {ckpt_path.name} does not match the model "
                f"architecture: {len(missing)} missing / {len(unexpected)} "
                f"unexpected weight keys.  The MODEL ARCHITECTURE CONFIG "
                f"likely drifted from the run that produced this checkpoint "
                f"(running anyway would generate garbage from partially-"
                f"random weights).  Reconcile the arch constants, or set "
                f"STRICT_CHECKPOINT_LOAD=False if this mismatch is known-benign."
            )
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
    """Pack a tricor supercell at the requested regime/density.

    Sub-stage timing is printed so an anomalous `pack=` time can be localized
    to read / shell_target / measure_g3 / generate without guessing.  The
    reference-derived steps (from_atoms, measure_g3) depend only on the CIF —
    if they dominate, they should be cached per-CIF (see _build_cif_cache in
    generate_mace_trajectories.py) rather than recomputed per trajectory.
    """
    from tricor import G3Distribution

    _t = time.perf_counter()
    ref = ase_read(str(cif_path), format="cif")
    t_read = time.perf_counter() - _t

    _t = time.perf_counter()
    shell = tc.CoordinationShellTarget.from_atoms(ref, phi_num_bins=90)
    t_shell = time.perf_counter() - _t

    _t = time.perf_counter()
    dist = G3Distribution(ref, label=str(cif_path.stem))
    dist.measure_g3(r_max=10.0, r_step=0.1, phi_num_bins=90, show_progress=False)
    t_g3 = time.perf_counter() - _t

    _t = time.perf_counter()
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
    t_gen = time.perf_counter() - _t

    print(f"    [pack-breakdown] ref_atoms={len(ref)}  "
          f"read={t_read:.1f}s  shell={t_shell:.1f}s  g3={t_g3:.1f}s  "
          f"generate={t_gen:.1f}s", flush=True)
    return cell, shell, ref, summary


def cleanup(cell, shell, device=None):
    # Pass through the GPU device — tricor's bond_relax dispatches to a
    # hybrid CPU-cKDTree + GPU-force-scatter path with pair-list caching
    # when device is non-None.  Drops bond_relax wall-clock from ~17 s →
    # ~4 s at 100×100×400 / 358 k atoms (TiO2), with sub-machine-
    # precision agreement vs the CPU path.
    cell.bond_relax(
        shell, n_iter=BOND_RELAX_N_ITER, max_step=BOND_RELAX_MAX_STEP,
        device=device,
    )


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


def _chunked_edge_processor_forward(ep, chunk_size: int):
    """Return a chunked drop-in for ``EdgeProcessor.forward``.

    Each edge's MLP is independent of other edges, so we can pre-
    allocate the output tensor and fill it in place per chunk —
    bit-exact to processing the full edge set in one shot.  The
    in-place fill avoids holding a list of per-chunk tensors that
    would otherwise sum to the full (E, edge_dim) anyway, and avoids
    the extra (E, edge_dim) intermediate that ``torch.cat`` of those
    chunks would allocate.  Memory profile:

      * cat input chunk      : (chunk_size, 2*node_dim + edge_dim)
      * MLP output chunk     : (chunk_size, edge_dim)
      * pre-alloc result     : (E, edge_dim)       ← one allocation
      * residual sum result  : (E, edge_dim)       ← at return only
    """

    def chunked_forward(x_i, x_j, edge_attr):
        E = edge_attr.shape[0]
        if chunk_size <= 0 or E <= chunk_size:
            out = torch.cat([x_i, x_j, edge_attr], dim=-1)
            out = ep.edge_mlp(out)
            return edge_attr + out

        out = torch.empty_like(edge_attr)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            ch = torch.cat(
                [x_i[start:end], x_j[start:end], edge_attr[start:end]],
                dim=-1,
            )
            out[start:end] = ep.edge_mlp(ch)
        return edge_attr + out

    return chunked_forward


def patch_model_edge_chunking(model, chunk_size: int) -> None:
    """Install chunked ``forward`` on every ``EdgeProcessor`` in the
    model so each conv layer's (E, 384) cat tensor never exceeds
    ``chunk_size``-many rows at once.  No-op when ``chunk_size <= 0``.
    """
    if chunk_size <= 0:
        return
    from graphite.nn.convs.mgn import EdgeProcessor
    for mod in model.modules():
        if isinstance(mod, EdgeProcessor):
            mod.forward = _chunked_edge_processor_forward(mod, chunk_size)


@torch.no_grad()
def run_iterative_inference(
    model, initial_positions, cell, species, weight_vector, shell_target,
    device, *, cutoff=CUTOFF, max_iter=MAX_ITER, tol=CONVERGENCE_TOL_ANG,
    collect_every=0,
):
    """Iteratively apply the student model until convergence or max_iter.

    Returns (final_positions, n_iter_run, intermediates, max_step_final_A).

    ``max_step_final_A`` is the largest per-atom displacement on the last
    completed iteration — convergence is ``max_step_final_A < tol``.
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

    # Detect model parameter dtype.  When USE_BF16_INFERENCE flipped the
    # model to bf16, all float input tensors must match — otherwise the
    # first Linear layer raises a dtype-mismatch error.  ``pos`` itself
    # stays in fp32 so the running position update doesn't compound bf16
    # rounding noise.  Only the model's float inputs (edge_attr, w,
    # shell features) and its output (``delta``) need conversion.
    _mdtype = next(model.parameters()).dtype
    _use_bf16 = (_mdtype == torch.bfloat16)
    if _use_bf16:
        # Shell features are static across iterations — cast once.
        # ``w_t`` is also static — cast once.
        w_t = w_t.bfloat16()
        shell["pair_features"] = shell["pair_features"].bfloat16()
        shell["trip_features"] = shell["trip_features"].bfloat16()

    intermediates: list[tuple[int, np.ndarray]] = []
    last_max_step: float = float("inf")

    def _record(iter_idx: int, p: torch.Tensor) -> None:
        intermediates.append((iter_idx, p.detach().cpu().numpy().copy()))

    for it in range(max_iter):
        batch = _build_data(pos, cell_t, z, w_t, shell, cutoff)
        if _use_bf16:
            # edge_attr was just computed in fp32 inside _build_data
            # from the fp32 ``pos`` (so the geometric graph stays
            # precise).  Cast it for the model forward only.
            batch.edge_attr = batch.edge_attr.bfloat16()
        delta = model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
            batch.shell_pair_species, batch.shell_pair_features,
            batch.shell_pair_batch,
            batch.shell_trip_species, batch.shell_trip_features,
            batch.shell_trip_batch,
        )
        if _use_bf16:
            delta = delta.float()
        d_norms = delta.norm(dim=-1)
        print(
            f"    iter {it:2d}: |delta| mean={d_norms.mean().item():.4f}  "
            f"max={d_norms.max().item():.4f}  Å"
        )
        pos_new = _wrap_positions(pos + delta, cell_t)
        max_step = (pos_new - pos).norm(dim=-1).max().item()
        last_max_step = float(max_step)
        pos = pos_new
        if collect_every > 0 and ((it + 1) % collect_every == 0):
            _record(it + 1, pos)
        if max_step < tol:
            if collect_every > 0 and (not intermediates or intermediates[-1][0] != it + 1):
                _record(it + 1, pos)
            return pos.cpu().numpy(), it + 1, intermediates, last_max_step
    if collect_every > 0 and (not intermediates or intermediates[-1][0] != max_iter):
        _record(max_iter, pos)
    return pos.cpu().numpy(), max_iter, intermediates, last_max_step


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
    # ── provenance / linkage ──────────────────────────────────────────────
    run_id:            str   = ""
    cif_idx:           int   = -1
    rel_density_target: float = -1.0
    actual_density_at_per_A3: float = -1.0
    # ── inference behaviour ───────────────────────────────────────────────
    converged:         bool  = False
    n_iter_max:        int   = -1
    max_step_final_A:  float = -1.0
    peak_gpu_gb:       float = -1.0
    # ── per-stage timings (seconds) ───────────────────────────────────────
    build_time_s:      float = -1.0
    cleanup_time_s:    float = -1.0
    setup_time_s:      float = -1.0
    inference_time_s:  float = -1.0
    save_time_s:       float = -1.0
    # ── cell shape — handy for downstream filtering without re-reading NPZ ─
    cell_a_A:          float = 0.0
    cell_b_A:          float = 0.0
    cell_c_A:          float = 0.0
    # ── error string (empty on success) ───────────────────────────────────
    error:             str   = ""


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
            run_id=np.asarray(traj.run_id),
            cif_idx=np.int64(traj.cif_idx),
            schema_version=np.int32(1),
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
            atoms.info["run_id"] = traj.run_id
            atoms.info["rng_seed"] = int(traj.rng_seed)
            atoms.info["cif_idx"] = int(traj.cif_idx)
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
                atoms.info["run_id"] = traj.run_id
                atoms.info["rng_seed"] = int(traj.rng_seed)
                atoms.info["cif_idx"] = int(traj.cif_idx)
                if it_idx is not None:
                    atoms.info["iter"] = int(it_idx)
                frames.append(atoms)
            ase_write(str(out_dir / f"{base}.xyz"), frames, format="extxyz")


# ─────────────────────────────────────────────────────────────────────────────
# Per-CIF orchestration
# ─────────────────────────────────────────────────────────────────────────────

def generate_for_cif(model, cif_path: Path, device,
                     *, run_id: str, cif_idx: int) -> list[GenResult]:
    """Generate one trajectory per (regime, seed) for this CIF.

    ``run_id`` + ``cif_idx`` are stamped on every produced trajectory so the
    flat dataset table can join back to ``_runs/{run_id}.json``.
    """
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
            # Reset peak-memory tracking so peak_gpu_gb reflects only this
            # trajectory, not the cumulative high-water-mark of the run.
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
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
                cleanup(cell, shell, device=device)
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
                final_pos, n_iter, intermediates, last_max_step = run_iterative_inference(
                    model, initial_pos, cell_arr, species_numbers,
                    weight_vector, shell_arrays, device,
                    collect_every=collect,
                )
                t_model = time.time() - _ts

                n_atoms = int(len(species_numbers))
                cell_volume = float(abs(np.linalg.det(cell_arr)))
                actual_density = (n_atoms / cell_volume) if cell_volume > 0 else -1.0
                peak_gb = -1.0
                if torch.cuda.is_available():
                    try:
                        peak_gb = float(
                            torch.cuda.max_memory_allocated(device) / 1e9
                        )
                    except Exception:
                        peak_gb = -1.0
                cell_diag = np.diag(cell_arr)

                traj = GenResult(
                    cif_filename=cif_path.name,
                    compound=compound,
                    mp_id=mp_id,
                    regime=regime,
                    rng_seed=seed,
                    n_atoms=n_atoms,
                    n_iter=int(n_iter),
                    runtime_sec=float(time.time() - t0),
                    out_npz="",
                    run_id=run_id,
                    cif_idx=int(cif_idx),
                    rel_density_target=float(rho),
                    actual_density_at_per_A3=float(actual_density),
                    converged=bool(last_max_step < CONVERGENCE_TOL_ANG),
                    n_iter_max=int(MAX_ITER),
                    max_step_final_A=float(last_max_step),
                    peak_gpu_gb=peak_gb,
                    build_time_s=float(t_pack),
                    cleanup_time_s=float(t_cleanup),
                    setup_time_s=float(t_setup),
                    inference_time_s=float(t_model),
                    cell_a_A=float(cell_diag[0]),
                    cell_b_A=float(cell_diag[1]),
                    cell_c_A=float(cell_diag[2]),
                )
                _ts = time.time()
                save_outputs(
                    out_dir, traj, initial_pos, final_pos, intermediates,
                    cell_arr, species_numbers, weight_vector, shell_arrays,
                    regime, rho, summary,
                )
                t_save = time.time() - _ts
                traj.save_time_s = float(t_save)
                # runtime_sec was captured before save; refresh to include it.
                traj.runtime_sec = float(time.time() - t0)
                results.append(traj)
                # Persist the row IMMEDIATELY so a crash before the next
                # trajectory finishes doesn't orphan this one.
                append_per_cif_manifest_row(out_dir, traj)
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
                fail_row = GenResult(
                    cif_filename=cif_path.name, compound=compound, mp_id=mp_id,
                    regime=regime, rng_seed=seed, n_atoms=-1, n_iter=0,
                    runtime_sec=time.time() - t0, out_npz="",
                    run_id=run_id, cif_idx=int(cif_idx),
                    rel_density_target=float(rho),
                    n_iter_max=int(MAX_ITER),
                    error=err,
                )
                results.append(fail_row)
                append_per_cif_manifest_row(out_dir, fail_row)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                traceback.print_exc(limit=3, file=sys.stdout)
                fail_row = GenResult(
                    cif_filename=cif_path.name, compound=compound, mp_id=mp_id,
                    regime=regime, rng_seed=seed, n_atoms=-1, n_iter=0,
                    runtime_sec=time.time() - t0, out_npz="",
                    run_id=run_id, cif_idx=int(cif_idx),
                    rel_density_target=float(rho),
                    n_iter_max=int(MAX_ITER),
                    error=err,
                )
                results.append(fail_row)
                append_per_cif_manifest_row(out_dir, fail_row)
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
        # Reconcile any stale 'running' manifests left behind by previous
        # runs that died ungracefully (SIGKILL / OOM-killer / node crash).
        # SLURM-aware: a still-running concurrent job is left alone.
        report = reconcile_stale_running_manifests()
        if report["scanned"] > 0:
            print(f"[stale_check] scanned={report['scanned']} "
                  f"marked_stale={report['marked']} "
                  f"left_running={report['left']}")

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

    # Every rank derives the same run_id from SLURM_JOB_ID + git SHA; only
    # rank 0 materializes the manifest file.  Each trajectory stamps run_id
    # so the flat dataset table can join back to the CONFIG snapshot.
    run_id = derive_run_id()
    run_manifest_path = OUTPUT_ROOT / "_runs" / f"{run_id}.json"
    if _GLOBAL_RANK == 0:
        write_run_manifest_start(run_id, ckpt_path)
        # Install interrupt + atexit handlers AFTER the manifest exists so
        # there's something for them to flip.  These catch graceful kills
        # (Ctrl+C, SLURM SIGTERM warning) — SIGKILL is handled by
        # reconcile_stale_running_manifests() on the next run.
        _install_interrupt_handlers(run_manifest_path)
        print(f"[run_id] {run_id}")
        print(f"[run_manifest] {run_manifest_path}")

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
    print(f"[rank {_GLOBAL_RANK}/{_WORLD_SIZE}] device={device}  "
          f"run_id={run_id}", flush=True)

    cif_paths_all = list_cifs()
    if not cif_paths_all:
        raise SystemExit(f"[abort] no CIFs to process under {CIF_DIR}")

    # Round-robin partition across workers — each rank handles cif_idx % N == rank.
    # Better than chunk-partition for uneven workloads (some CIFs are bigger).
    # Carry the global cif_idx alongside the path so the per-traj row can
    # record it (this is the same index used by seed = BASE_SEED + cif_idx
    # * SEED_STEP + traj_idx in the cleanup tooling, so it has to be the
    # CORPUS-WIDE index, not the per-rank slice index).
    cif_pairs_all = list(enumerate(cif_paths_all))
    cif_pairs = cif_pairs_all[_GLOBAL_RANK::_WORLD_SIZE]

    n_per_cif = len(REGIMES) * len(SEEDS)
    if _GLOBAL_RANK == 0:
        print(f"[corpus] {len(cif_paths_all)} CIFs total × {len(REGIMES)} regimes "
              f"× {len(SEEDS)} seeds = {len(cif_paths_all) * n_per_cif} trajectories")
        print(f"[output] root   = {OUTPUT_ROOT}")
    print(f"[rank {_GLOBAL_RANK}/{_WORLD_SIZE}] my slice: "
          f"{len(cif_pairs)} CIFs × {n_per_cif} = "
          f"{len(cif_pairs) * n_per_cif} trajectories", flush=True)
    print()

    model = _load_model(ckpt_path, device)
    # Install edge chunking on every MeshGraphNetsConv's EdgeProcessor
    # to keep the (E, 384) cat tensor at most EDGE_CHUNK_SIZE rows at
    # once.  Bit-exact to the unchunked forward; drops peak GPU memory
    # from ~93 GB → ~40 GB at Fe2N 100×100×400 (30 M edges).
    patch_model_edge_chunking(model, EDGE_CHUNK_SIZE)
    print(f"[edge_chunking] EDGE_CHUNK_SIZE={EDGE_CHUNK_SIZE:,}", flush=True)
    # bf16 conversion BEFORE torch.compile so dynamo traces bf16 ops.
    # Casts all model weights to bf16; LayerNorm internally still
    # computes in fp32 for numerical stability (PyTorch default).
    if USE_BF16_INFERENCE and device.type == "cuda":
        model = model.bfloat16()
        print(f"[bf16] model converted to bfloat16 — "
              f"A100 Tensor Cores active", flush=True)
    elif USE_BF16_INFERENCE:
        # bf16 only buys speed on GPU Tensor Cores; on CPU it offers no
        # speedup and risks slow/unsupported kernels.  Stay in fp32 so a
        # CPU-only run still works.  ``_use_bf16`` downstream keys off the
        # model's actual dtype, so leaving it fp32 keeps the run consistent.
        print(f"[bf16] device={device.type}: skipping bf16 cast "
              f"(GPU-only optimization), running fp32", flush=True)
    # ``torch.compile`` MUST come AFTER the edge-chunking patch so
    # dynamo traces the chunked forward (otherwise the patched
    # ``EdgeProcessor.forward`` runs eager while the rest is compiled).
    if USE_TORCH_COMPILE:
        print(f"[torch.compile] mode={TORCH_COMPILE_MODE} dynamic=True "
              f"— first call will incur ~10-20 s compile cost",
              flush=True)
        model = torch.compile(
            model, mode=TORCH_COMPILE_MODE, dynamic=True,
        )

    all_results: list[GenResult] = []
    t_start = time.time()
    for i, (cif_idx, cif_path) in enumerate(cif_pairs, 1):
        print(f"[{i}/{len(cif_pairs)}] (cif_idx={cif_idx}) {cif_path.name}")
        rs = generate_for_cif(
            model, cif_path, device, run_id=run_id, cif_idx=cif_idx,
        )
        all_results.extend(rs)
        elapsed = time.time() - t_start
        rate = i / max(elapsed, 1e-9) * 60.0
        eta_sec = (len(cif_pairs) - i) * (elapsed / max(i, 1))
        print(f"  cumulative: {len(all_results)} trajs done  "
              f"rate={rate:.1f} CIF/min  ETA={int(eta_sec/60)} min")
        print()

    # ── Per-rank summary CSV (replaces the previously racy shared write) ──
    # Each rank writes its own _runs/{run_id}.rank{N}.csv.  The flat-table
    # builder script joins all rank summaries + the run JSON into the
    # corpus-wide table.  No barrier needed — each rank is self-contained.
    if all_results:
        rank_csv = write_per_rank_summary(run_id, all_results)
        print(f"[summary] rank {_GLOBAL_RANK} wrote {rank_csv}")

    # ── Final breakdown ──────────────────────────────────────────────────
    n_ok = sum(1 for r in all_results if not r.error)
    n_fail = len(all_results) - n_ok
    wall_clock_min = (time.time() - t_start) / 60.0
    print()
    print("=" * 60)
    print(f"  Total trajectories : {len(all_results)}")
    print(f"  Succeeded          : {n_ok}")
    print(f"  Failed             : {n_fail}")
    print(f"  Total wall-clock   : {wall_clock_min:.1f} min")
    print(f"  Output root        : {OUTPUT_ROOT}")
    print("=" * 60)

    # ── Update run manifest at end (rank 0 only) ─────────────────────────
    # Stats here are rank-0-local; full corpus-wide stats are computed
    # later by build_dataset_table.py once all per-rank summaries are
    # available.  We still snapshot rank-0's view so a running-then-
    # killed run still produces something useful.
    if _GLOBAL_RANK == 0:
        update_run_manifest_end(
            run_manifest_path,
            status="completed",
            stats={
                "rank0_n_trajectories": len(all_results),
                "rank0_n_succeeded":    n_ok,
                "rank0_n_failed":       n_fail,
                "rank0_wall_clock_min": float(wall_clock_min),
                "world_size":           _WORLD_SIZE,
                "n_cifs_in_corpus":     len(cif_paths_all),
            },
        )


if __name__ == "__main__":
    main()
