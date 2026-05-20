"""Train the relaxml surrogate, v2 (per-edge shell_target injection) variant.

Variant of ``shelltgt_phys/train.py`` that uses
``tricor.relaxml.shelltgt_phys_v2``: pair shell_target features feed
the edge encoder directly instead of going through a per-graph
deep-set encoder.  See ``src/tricor/relaxml/shelltgt_phys_v2/model.py``
for the rationale.

Data layer is unchanged — re-uses ``RelaxMLDataModule`` from
``tricor.relaxml.data_shelltgt``.  Manifests / .npz files produced for
v1 training work as-is.

Before running: ``nvtop`` / ``nvidia-smi`` to see which GPU is free,
set ``GPU_ID`` below, then:

    python train.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (MUST be before numpy/torch imports)
# ─────────────────────────────────────────────────────────────────────────────

# --- buffle resource caps ---
GPU_ID = 2
NUM_THREADS = 4
CHECK_GPU_BUSY = False

# --- data ---
# Manifest pointing at trajectories for Si + SiC + SiO2 + BN + AlN (or
# whatever cross-composition training set you've assembled).  v2 .npz
# schema is identical to v1, so existing manifests work.
from pathlib import Path
MANIFEST = str(
    Path(__file__).parent.parent
    / "data/sio2_polymorphs_v1/merged_train_for_stishovite/manifest.csv"
)

CUTOFF = 5.0
K_STRIDE_SNAPSHOTS = 1
ROTATE = True
VAL_FRACTION = 0.1
SPLIT_SEED = 42

# --- model ---
MAX_Z = 120
NODE_DIM = 128
EDGE_DIM = 128
NUM_CONVS = 4
WEIGHT_ENCODER_HIDDEN = 64
SPECIES_PAIR_DIM = 16
SPECIES_HIDDEN = 128
SHELL_TARGET_SPECIES_DIM = 8
SHELL_TARGET_HIDDEN = 64
# CFG dropout on shell_target conditioning.  In v1 we raised this from
# 0.2 -> 0.4 trying to recover from the deep-set encoder going inert;
# v2 sidesteps that pathology structurally (per-edge features have no
# learnable layer that can collapse to zero), so 0.2 is a reasonable
# starting point.  Lower it further to 0.0 if you don't care about
# inference-time CFG / "no shell_target" graceful degradation.
SHELL_TARGET_DROPOUT = 0.2
# Auxiliary bond-length loss weight.  When > 0, training adds
# ``AUX_BOND_WEIGHT * mean(((‖pred_post_step_distance‖ - target_r) /
# sigma)²)`` over edges whose species pair has a shell_target entry.
# Directly supervises the model to USE the per-edge target_r feature,
# fixing the v2 pair-slope-≈0 failure mode (the per-edge channel is
# structurally present but downstream layers learn tiny weights on it
# under the relaxation MSE alone).
#
# Magnitudes: the displacement MSE is in Å² (typical ~1e-3 Å² per
# atom-step); aux loss is in σ² units (typical ~1 when the model is
# wrong, ~0.1 when right).  AUX_BOND_WEIGHT=0.01 makes the aux
# contribution comparable to the displacement loss early in training;
# bump to 0.1 for aggressive supervision; 0.0 disables.
AUX_BOND_WEIGHT = 0.1
EMA_DECAY = 0.9999
LR = 1e-3
LR_SCHEDULE = "cosine"
LR_MIN_RATIO = 0.01
WARMUP_STEPS = 500

# --- training ---
MAX_EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 4
LOG_DIR = "./lightning_logs"
RUN_NAME = "coord_phys_rinject"
# v2 has a different parameter set than v1 (different module names,
# edge_encoder.embed first-layer has 5 extra input channels) so v1
# checkpoints can't be loaded.  Set to a v2 checkpoint to resume.
RESUME_CKPT = None

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

from tricor.relaxml.shelltgt_phys_v2 import LitRelaxML
from tricor.relaxml.data_shelltgt import RelaxMLDataModule


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
    if used_mb > 1024:
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
        species_hidden=SPECIES_HIDDEN,
        shell_target_species_dim=SHELL_TARGET_SPECIES_DIM,
        shell_target_hidden=SHELL_TARGET_HIDDEN,
        shell_target_dropout=SHELL_TARGET_DROPOUT,
        aux_bond_weight=AUX_BOND_WEIGHT,
        ema_decay=EMA_DECAY,
        learn_rate=LR,
        lr_schedule=LR_SCHEDULE,
        lr_min_ratio=LR_MIN_RATIO,
        warmup_steps=WARMUP_STEPS,
    )

    # torch.compile the inner model for ~1.5–2x throughput on Ampere/Hopper.
    # build_per_edge_shell_target and TripletTargetEncoder.forward stay in
    # eager via @_dynamo.disable so variable per-pair / per-triplet shapes
    # don't trigger recompiles.
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
