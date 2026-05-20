"""Train the shell_target-conditioned relaxml surrogate, physics-feature variant.

Variant of ``train_shelltgt.py`` that uses the model from
``tricor.relaxml.shelltgt_phys`` (physics-feature SpeciesEncoder in
place of nn.Embedding lookups).  Data layer is unchanged — re-uses the
same RelaxMLDataModule from ``tricor.relaxml.data_shelltgt``.

Test design (see RELAXML_SESSION.txt and the synthesis report):
  * Train on Si + SiC + SiO2 + BN + AlN  (~150 trajectories per compound)
  * Held-out test on Si3N4               (~30 trajectories)
  The model has seen Si and N individually but never the (Si, N) pair.
  Compare per-pair g(r) and ADF MAE against the baseline
  ``train_shelltgt.py`` checkpoint trained on the same data.

Before running: ``nvtop`` / ``nvidia-smi`` to see which GPUs are free,
set ``GPU_IDS`` below (one entry = single-GPU, multiple = DDP), then:

    python train.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (MUST be before numpy/torch imports)
# ─────────────────────────────────────────────────────────────────────────────

# --- mallard resource caps ---
# Set ``GPU_IDS`` to a list of physical GPU indices.  One entry = single-GPU
# training; two or more = DDP across that many ranks.  Each rank holds an
# independent dataloader pool of ``NUM_WORKERS`` workers, so total worker
# processes = NUM_WORKERS * len(GPU_IDS) — watch system RAM on the shared
# box (see RELAXML_SESSION.txt §19.6).  Effective batch per optimizer step
# = BATCH_SIZE * len(GPU_IDS); Adam-style sqrt LR scaling means if you go
# 1 → N GPUs at fixed BATCH_SIZE, rescale LR by sqrt(N).
GPU_IDS = [2, 3]
NUM_THREADS = 4
CHECK_GPU_BUSY = False

# --- data ---
# Manifest pointing at trajectories for Si + SiC + SiO2 + BN + AlN (or
# whatever cross-composition training set you've assembled).  See
# RELAXML_PHYS_NEXTSTEPS.md for the merge_manifests.py recipe.
from pathlib import Path
MANIFEST = str(
    Path(__file__).parent.parent
    / "/wigeon/users/ehrdt/prod/relaxml_big_v1/manifest.csv" #"data/si-n-trajectories/manifest.csv"
)

CUTOFF = 5.0
K_STRIDE_SNAPSHOTS = 1
ROTATE = True
VAL_FRACTION = 0.1
SPLIT_SEED = 42

# Max trajectories held in each dataloader worker's LRU cache.  Each
# entry is ~5-10 MB.  Total dataloader cache budget = CACHE_MAX_ENTRIES
# * NUM_WORKERS * len(GPU_IDS).  At 500 × 8 × 2 = 8000 entries ≈ 50 GB
# (with ~7 MB avg).  Set None for unbounded; on the 10k-trajectory big
# dataset that blew up to ~880 GB.  Drop to 200-300 if RAM is tight.
CACHE_MAX_ENTRIES = 200

# --- model ---
MAX_Z = 120
NODE_DIM = 128
EDGE_DIM = 128
NUM_CONVS = 4
WEIGHT_ENCODER_HIDDEN = 64
SPECIES_PAIR_DIM = 16
# SpeciesEncoder hidden width.  128 is plenty for the 39-D periodic
# table input; deeper/wider tends to overfit on 120 element rows.
SPECIES_HIDDEN = 128
SHELL_TARGET_SPECIES_DIM = 8
SHELL_TARGET_HIDDEN = 64
SHELL_TARGET_DROPOUT = 0.2

# Auxiliary bond-length loss (shell_target conditioning aux supervision).
# Closed under the §13.11 scope decision — shell_target is downgraded
# to auxiliary metadata.  Leave at 0.0; the lookup is still in the
# codebase for posterity but not active in this experiment.
AUX_BOND_WEIGHT = 0.0

EMA_DECAY = 0.9999
LR = 1.5e-3
LR_SCHEDULE = "cosine"
LR_MIN_RATIO = 0.01
WARMUP_STEPS = 500

# --- training ---
MAX_EPOCHS = 100
BATCH_SIZE = 6
# Per-rank worker count.  Total worker processes = NUM_WORKERS *
# len(GPU_IDS).  Single-GPU + 12 workers saturated the GPU at 90+%
# util; dropped to 8/rank here so 2-GPU DDP totals 16 (vs 24) and
# keeps system RAM headroom on the shared box — see §19.6.  Bump
# back up if profiling shows dataloader starvation.
NUM_WORKERS = 8
LOG_DIR = "./lightning_logs"
RUN_NAME = "516_struct"
RESUME_CKPT = None # "./lightning_logs/si-n-phys_mitigated/version_5/checkpoints/last.ckpt"

# ─────────────────────────────────────────────────────────────────────────────
# Apply resource caps BEFORE importing numpy/torch
# ─────────────────────────────────────────────────────────────────────────────

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)

# Variable batch shapes (from variable atom counts per cell) cause the
# CUDA caching allocator to retain many fixed-size segments without
# being able to coalesce them — observed as 85 GiB reserved at only
# 31 GiB peak active.  ``expandable_segments:True`` (PyTorch 2.1+) uses
# one growable virtual address range that expands/contracts as needed,
# eliminating most fragmentation for variable-shape workloads.  Must be
# set BEFORE the CUDA context initializes, so before ``import torch``.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

from tricor.relaxml.shelltgt_phys import LitRelaxML
from tricor.relaxml.data_shelltgt import RelaxMLDataModule


class MemorySummaryCallback(L.Callback):
    """Print ``torch.cuda.memory_summary()`` once, at the end of the first
    training epoch.

    Diagnostic for the 80+ GiB VRAM usage at small batch sizes — the
    summary breaks down allocator state (reserved vs active, allocation
    histogram, retries) so we can see whether activations, the inductor
    compile cache, or something else dominates.  One-shot via the
    ``_printed`` flag so it doesn't spam every epoch.
    """

    def __init__(self) -> None:
        super().__init__()
        self._printed = False

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: D401
        if self._printed or not torch.cuda.is_available():
            return
        if not trainer.is_global_zero:
            return
        self._printed = True
        sep = "=" * 78
        print(f"\n{sep}\nCUDA memory summary after epoch {trainer.current_epoch}:\n{sep}")
        print(torch.cuda.memory_summary(abbreviated=False))
        print(sep + "\n", flush=True)


def check_gpu_availability(gpu_ids: list[int]) -> None:
    """Abort if any chosen GPU already has another job running."""
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

    busy_mb: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx_str, mem_str = [x.strip() for x in line.split(",")]
        busy_mb[int(idx_str)] = int(mem_str)

    bad: list[tuple[int, int]] = []
    for g in gpu_ids:
        if g not in busy_mb:
            print(f"[warn] GPU {g} not found in nvidia-smi output; skipping.")
            continue
        if busy_mb[g] > 1024:
            bad.append((g, busy_mb[g]))
    if bad:
        for g, mb in bad:
            print(
                f"[abort] GPU {g} already has {mb} MiB in use — another job "
                f"may be running.  Pick different GPU_IDS or set "
                f"CHECK_GPU_BUSY=False.",
            )
        sys.exit(1)
    print(f"[ok] GPUs {gpu_ids} look idle.")


def main() -> None:
    if CHECK_GPU_BUSY:
        check_gpu_availability(GPU_IDS)

    dm = RelaxMLDataModule(
        manifest_path=MANIFEST,
        cutoff=CUTOFF,
        k_stride_snapshots=K_STRIDE_SNAPSHOTS,
        rotate=ROTATE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_fraction=VAL_FRACTION,
        split_seed=SPLIT_SEED,
        cache_max_entries=CACHE_MAX_ENTRIES,
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
    # ShellTargetEncoder.forward stays in eager via @_dynamo.disable so
    # variable per-pair / per-triplet shapes don't trigger recompiles.
    # SpeciesEncoder runs inside torch.compile (the periodic-table buffer
    # has a fixed shape, so no recompilation pressure from there).
    #
    # mode="default" (with dynamic=True) keeps inductor's kernel
    # optimizations (op fusion, code-gen) without the CUDA-graph
    # machinery.  mode="reduce-overhead" would record a new CUDA graph
    # per unique batch shape and bloat VRAM; the skip-dynamic-graphs
    # flag that's supposed to cap that hits an internal PyTorch
    # assertion in the backward pass (manager is None) when combined
    # with dynamic=True.  Proper fix would be shape-bucketing/padding
    # in the dataloader — deferred until it's needed.
    lit.model = torch.compile(lit.model, mode="default", dynamic=True)

    logger = TensorBoardLogger(save_dir=LOG_DIR, name=RUN_NAME)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        MemorySummaryCallback(),
        ModelCheckpoint(
            monitor="val_loss", mode="min",
            save_top_k=3, save_last=True,
            filename="{epoch:03d}-{val_loss:.4f}",
        ),
        TQDMProgressBar(refresh_rate=20),
    ]

    # DDP when GPU_IDS has >1 entry.  Lightning auto-wraps shuffle=True
    # dataloaders with a DistributedSampler.  Gradients are all-reduced
    # after backward; since the EMA update runs after super().optimizer_step()
    # (i.e. after the grad sync), every rank steps from identical params
    # and EMAs stay in sync across ranks too.
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=len(GPU_IDS),
        strategy="ddp" if len(GPU_IDS) > 1 else "auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=25,
        # Val accounts for ~10-15% of each "epoch" wall time.  Running
        # it every 3rd epoch saves that fraction without affecting
        # convergence — ModelCheckpoint still triggers on val_loss, just
        # less often.
        # Lowered from 3 → 1 while big-dataset runs are still
        # OOM-prone; we want a checkpoint after epoch 1 rather than
        # losing 3 epochs on the next crash.  Bump back to 3 once a
        # run has cleared the first ~5 epochs without OOM.
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
    )
    trainer.fit(lit, datamodule=dm, ckpt_path=RESUME_CKPT)


if __name__ == "__main__":
    main()
