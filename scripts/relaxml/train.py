"""Train the relaxml step-by-step surrogate (mallard-aware variant).

Follows the mallard shared-server usage guidelines:
  * Caps CPU threads (OMP_NUM_THREADS, torch.set_num_threads) — must be set
    before importing numpy/torch so the library init picks them up.
  * Pins the process to a single, user-selected GPU via CUDA_VISIBLE_DEVICES.
  * Checks ``nvidia-smi`` at launch so you don't stomp on another user's job.

Before running: check ``nvtop`` / ``nvidia-smi`` to see which GPU is free,
set ``GPU_ID`` below accordingly, then:

    python train.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (MUST be before numpy/torch imports)
# ─────────────────────────────────────────────────────────────────────────────

# --- mallard resource caps ---
GPU_ID = 2                # physical GPU index (check with nvidia-smi / nvtop)
NUM_THREADS = 4           # CPU threads; match NUM_WORKERS below.
CHECK_GPU_BUSY = True     # abort if the chosen GPU already has another job

# --- data ---
MANIFEST = "./data/multi_species_v1/merged_subset_Si/manifest.csv" #"./data/multicomp_trajectories_merged/manifest.csv"
CUTOFF = 5.0              # Å
K_STRIDE_SNAPSHOTS = 1    # consecutive snapshots (= 5 tricor steps per model step)
ROTATE = True             # SO(3) augmentation at training time
VAL_FRACTION = 0.1
SPLIT_SEED = 42

# --- model ---
MAX_Z = 120               # nn.Embedding table size; covers full periodic table
NODE_DIM = 128
EDGE_DIM = 128
NUM_CONVS = 4
WEIGHT_ENCODER_HIDDEN = 64
SPECIES_PAIR_DIM = 16     # edge-side species embedding (kept small)
EMA_DECAY = 0.9999
LR = 1e-3
LR_SCHEDULE = "cosine"    # "none" | "cosine"
LR_MIN_RATIO = 0.01
WARMUP_STEPS = 500

# --- training ---
MAX_EPOCHS = 100
BATCH_SIZE = 2            # graphs per batch; each graph ~6k atoms at 50 Å
NUM_WORKERS = 4           # DataLoader processes (match NUM_THREADS above)
LOG_DIR = "./lightning_logs"
RUN_NAME = "relaxml-subset-multi-si-real"
RESUME_CKPT = None        # path to .ckpt or None

# ─────────────────────────────────────────────────────────────────────────────
# Apply resource caps BEFORE importing numpy/torch
# ─────────────────────────────────────────────────────────────────────────────

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

_n = str(NUM_THREADS)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = _n

import subprocess
import sys

import torch
torch.set_num_threads(NUM_THREADS)

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelCheckpoint, TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from tricor.relaxml import LitRelaxML, RelaxMLDataModule


def check_gpu_availability(gpu_id: int) -> None:
    """Abort if another user's job is already on the chosen GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             f"--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception as e:
        print(f"[warn] could not query nvidia-smi ({e}); skipping GPU busy-check.")
        return

    used_mb = None
    for line in out.strip().splitlines():
        idx_str, mem_str = [x.strip() for x in line.split(",")]
        if int(idx_str) == gpu_id:
            used_mb = int(mem_str)
            break
    if used_mb is None:
        print(f"[warn] GPU {gpu_id} not found in nvidia-smi output; skipping.")
        return
    if used_mb > 1024:  # > 1 GiB is not idle
        print(
            f"[abort] GPU {gpu_id} already has {used_mb} MiB in use — another job "
            f"may be running.  Pick a different GPU_ID or set CHECK_GPU_BUSY=False.",
        )
        sys.exit(1)
    print(f"[ok] GPU {gpu_id} looks idle ({used_mb} MiB used).")


def main() -> None:
    if CHECK_GPU_BUSY:
        check_gpu_availability(GPU_ID)

    dm = RelaxMLDataModule(
        manifest_path=MANIFEST,
        cutoff=CUTOFF,
        k_stride_snapshots=K_STRIDE_SNAPSHOTS,
        rotate=ROTATE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_fraction=VAL_FRACTION,
        split_seed=SPLIT_SEED,
    )

    lit = LitRelaxML(
        max_z=MAX_Z,
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        num_convs=NUM_CONVS,
        weight_encoder_hidden=WEIGHT_ENCODER_HIDDEN,
        species_pair_dim=SPECIES_PAIR_DIM,
        ema_decay=EMA_DECAY,
        learn_rate=LR,
        lr_schedule=LR_SCHEDULE,
        lr_min_ratio=LR_MIN_RATIO,
        warmup_steps=WARMUP_STEPS,
    )

    # torch.compile the inner model for ~1.5–2x throughput on Ampere/Hopper.
    # dynamic=True so we don't pay recompilation cost when edge counts vary
    # slightly between batches (PBC neighbor lists drift with atom positions).
    # The first epoch will be slower than steady-state due to compilation.
    lit.model = torch.compile(lit.model, mode="default", dynamic=True)

    logger = TensorBoardLogger(save_dir=LOG_DIR, name=RUN_NAME)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss", mode="min",
            save_top_k=3, save_last=True,
            filename="{epoch:03d}-{val_loss:.4f}",
        ),
        TQDMProgressBar(refresh_rate=20),
    ]

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=25,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
    )
    trainer.fit(lit, datamodule=dm, ckpt_path=RESUME_CKPT)


if __name__ == "__main__":
    main()
