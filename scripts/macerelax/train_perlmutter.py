"""Multi-GPU adaptation of train.py for NERSC Perlmutter.

Trains the MACE+wall pilot model on a single Perlmutter A100 80 GB node
using all 4 GPUs via Lightning's DDP strategy.  Doesn't depend on the
mace/ utility dir (only the NPZ trajectories + the tricor.macerelax
package); safe to run with just the dataset on $SCRATCH and the conda
env on Perlmutter.

Differences from the workstation train.py:
  * GPU_ID → NUM_GPUS (default 4).  Lightning's "ddp" strategy assigns one
    GPU per rank automatically; we don't pin CUDA_VISIBLE_DEVICES.
  * CHECK_GPU_BUSY removed (SLURM allocations own the node).
  * MANIFEST / EXPERIMENTS_REGISTRY / LOG_DIR pointed at $SCRATCH paths.
  * NUM_WORKERS bumped — Perlmutter A100 nodes have 64 CPU cores so each
    GPU's dataloader can comfortably use 8 worker processes.
  * Effective batch size is BATCH_SIZE × NUM_GPUS × NUM_NODES.  Keeping
    BATCH_SIZE=8 with 4 GPUs → effective batch 32 per optimizer step.
  * Optional NUM_NODES knob for future multi-node scaling.  Default 1.

Launch path (single node, 4 GPUs):
    sbatch scripts/macerelax/train_perlmutter.sbatch

For a quick interactive smoke (validate the env + DDP launch):
    salloc -A m5241 -C "gpu&hbm80g" -q interactive -t 0:30:00 --gpus=4
    module load conda && conda activate tricor
    srun --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 \
        /global/common/software/m5020/ehrdt/tricor/bin/python \
        scripts/macerelax/train_perlmutter.py

NB: under DDP, val_loss must be reduced across ranks for EarlyStopping +
ModelCheckpoint to behave correctly.  Verify LitRelaxML's val_step calls
self.log("val_loss", ..., sync_dist=True).  If it doesn't, monitored
metrics will be rank-0-only and the checkpoints/early-stop will see only
that rank's batch — usable but suboptimal.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (MUST be before numpy/torch imports)
# ─────────────────────────────────────────────────────────────────────────────

# --- compute resources ---
# Number of GPUs per node.  Set to 4 for a full Perlmutter A100 node; can
# be reduced for debug-QoS testing.
NUM_GPUS = 4
# Number of nodes.  Keep at 1 unless you've set up a multi-node sbatch.
# Multi-node requires extra srun config (--ntasks-per-node=NUM_GPUS,
# Lightning's SLURMEnvironment plugin auto-detects) but the script
# changes are minimal beyond bumping this knob.
NUM_NODES = 1
# CPU threads per dataloader worker process.  With 4 GPUs × 8 workers each
# = 32 worker processes per node; 2 threads each = 64 total (= core count).
NUM_THREADS = 2

# --- data ---
# Adjust to your Perlmutter paths.  These should be on $SCRATCH for fast
# I/O during training (the manifests reference NPZs by relative path,
# resolved against the manifest's parent dir).
EXPERIMENTS_REGISTRY = "/pscratch/sd/e/ehrdt/macerelax/big_v1/manifests/experiments.json"
EXPERIMENT_NAME = "big_v1_full"   # name from make_experiment_manifests.py registry

CUTOFF = 5.0              # Å — periodic-radius-graph cutoff
K_STRIDE_SNAPSHOTS = 5    # FIRE steps between training (frame_t, frame_t+stride)
ROTATE = True             # SO(3) augmentation at training time
VAL_FRACTION = 0.1
SPLIT_SEED = 42

# --- model architecture (matches train.py — change only for arch sweeps) ---
MAX_Z = 120
NODE_DIM = 128
EDGE_DIM = 128
NUM_CONVS = 4
WEIGHT_ENCODER_HIDDEN = 64
SPECIES_PAIR_DIM = 16
SHELL_TARGET_SPECIES_DIM = 8
SHELL_TARGET_HIDDEN = 64
SHELL_TARGET_DROPOUT = 0.0
EMA_DECAY = 0.9999

# --- optimization ---
LR = 1e-3                 # base learning rate; cosine schedule handles
                           # the implicit scaling for multi-GPU effective
                           # batch in most cases.  If training is unstable,
                           # consider LR / sqrt(NUM_GPUS).
LR_SCHEDULE = "cosine"
LR_MIN_RATIO = 0.01
WARMUP_STEPS = 500
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE_EPOCHS = 10

# --- training ---
MAX_EPOCHS = 100
# Per-GPU batch size.  Effective batch per optimizer step =
# BATCH_SIZE * NUM_GPUS * NUM_NODES = 8 * 4 * 1 = 32 for the default.
BATCH_SIZE = 8
# DataLoader processes PER RANK.  Total processes per node = NUM_WORKERS
# * NUM_GPUS = 8 * 4 = 32, leaving headroom on 64-core Perlmutter A100
# nodes.  Each holds an LRU cache of trajectories; total cache memory
# could be ~10 GB at default cache sizes — well under available RAM.
NUM_WORKERS = 8

LOG_DIR = "/pscratch/sd/e/ehrdt/macerelax/lightning_logs"
RUN_NAME = "big_v1_full_stride5_v1"   # tensorboard + checkpoint subdir name
RESUME_CKPT = None        # path to .ckpt to resume training from, or None

# ─────────────────────────────────────────────────────────────────────────────
# Apply resource caps BEFORE importing numpy/torch
# ─────────────────────────────────────────────────────────────────────────────

import os

# DDP needs all GPUs visible.  No CUDA_VISIBLE_DEVICES pin here.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_n = str(NUM_THREADS)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _n)

import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(NUM_THREADS)


def resolve_experiment(experiment_name: str, registry_path: str) -> dict:
    """Look up an experiment in the registry written by
    scripts/macerelax/generation/make_experiment_manifests.py.
    """
    path = Path(registry_path)
    with open(path) as f:
        reg = json.load(f)
    exps = reg.get("experiments", {})
    if experiment_name not in exps:
        raise SystemExit(
            f"[abort] experiment {experiment_name!r} not in registry "
            f"{registry_path}.  Available: {sorted(exps)}.  "
            f"Re-run scripts/macerelax/generation/make_experiment_manifests.py "
            f"after editing its EXPERIMENTS config to add new entries."
        )
    e = exps[experiment_name]
    base = path.parent
    return {
        "experiment_name":      experiment_name,
        "manifest":             str((base / e["train_manifest"]).resolve()),
        "eval_manifest":        str((base / e["eval_manifest"]).resolve()),
        "description":          e["description"],
        "train_systems":        list(e["train_systems"]),
        "eval_systems":         list(e["eval_systems"]),
        "n_train_trajectories": int(e["n_train_trajectories"]),
        "n_eval_trajectories":  int(e["n_eval_trajectories"]),
        "registry_generated_at": reg.get("generated_at", ""),
    }


import lightning as L
from lightning.pytorch.callbacks import (
    Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger


class ExperimentMetadataLogger(Callback):
    """At train start, write experiment.json into the version_dir for
    audit-trail durability (survives later EXPERIMENTS config edits)."""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload

    def on_train_start(self, trainer: L.Trainer,
                        pl_module: L.LightningModule) -> None:
        # Only rank 0 writes to avoid clobbering under DDP.
        if not trainer.is_global_zero:
            return
        log_dir = trainer.log_dir
        if log_dir is not None:
            out = Path(log_dir) / "experiment.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(self.payload, f, indent=2, default=str)
                f.write("\n")
            print(f"[experiment] metadata written to {out}")
        if trainer.logger is not None:
            flat = {
                k: (",".join(v) if isinstance(v, (list, tuple)) else v)
                for k, v in self.payload.items()
            }
            trainer.logger.log_hyperparams(flat)


from tricor.macerelax.model_perlmutter import LitRelaxML   # DDP-aware variant
from tricor.macerelax.data import RelaxMLDataModule


def main() -> None:
    exp = resolve_experiment(EXPERIMENT_NAME, EXPERIMENTS_REGISTRY)
    run_name = RUN_NAME or EXPERIMENT_NAME

    # Only rank 0 prints to avoid duplicated startup output under DDP.
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank == 0:
        print(f"[experiment] {exp['experiment_name']}: "
              f"{exp['n_train_trajectories']} train trajectories"
              f"  (held out: {exp['eval_systems']})")
        print(f"  train manifest : {exp['manifest']}")
        print(f"  eval  manifest : {exp['eval_manifest']}")
        print(f"  NUM_NODES={NUM_NODES}  NUM_GPUS={NUM_GPUS}  "
              f"BATCH_SIZE={BATCH_SIZE}  "
              f"(effective batch = {NUM_GPUS * NUM_NODES * BATCH_SIZE})")
        print(f"  RUN_NAME={run_name}  LOG_DIR={LOG_DIR}")

    dm = RelaxMLDataModule(
        manifest_path=exp["manifest"],
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
        shell_target_species_dim=SHELL_TARGET_SPECIES_DIM,
        shell_target_hidden=SHELL_TARGET_HIDDEN,
        shell_target_dropout=SHELL_TARGET_DROPOUT,
        ema_decay=EMA_DECAY,
        learn_rate=LR,
        lr_schedule=LR_SCHEDULE,
        lr_min_ratio=LR_MIN_RATIO,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
    )

    # torch.compile is compatible with DDP since PyTorch 2.x.  dynamic=True
    # avoids recompilation on slightly-different edge counts between batches.
    # First epoch will be slower than steady-state due to compile + DDP
    # synchronization warmup.
    lit.model = torch.compile(lit.model, mode="default", dynamic=True)

    experiment_payload = {
        "experiment_name":       exp["experiment_name"],
        "experiment_desc":       exp["description"],
        "manifest":              exp["manifest"],
        "eval_manifest":         exp["eval_manifest"],
        "train_systems":         exp["train_systems"],
        "eval_systems":          exp["eval_systems"],
        "n_train_trajectories":  exp["n_train_trajectories"],
        "n_eval_trajectories":   exp["n_eval_trajectories"],
        "registry_generated_at": exp["registry_generated_at"],
        "split_seed":            SPLIT_SEED,
        "val_fraction":          VAL_FRACTION,
        "k_stride_snapshots":    K_STRIDE_SNAPSHOTS,
        "cutoff":                CUTOFF,
        "rotate":                ROTATE,
        "shell_target_dropout":  SHELL_TARGET_DROPOUT,
        "weight_decay":          WEIGHT_DECAY,
        "early_stop_patience":   EARLY_STOP_PATIENCE_EPOCHS,
        # Multi-GPU context — useful for cross-run comparisons.
        "num_nodes":             NUM_NODES,
        "num_gpus":              NUM_GPUS,
        "batch_size_per_gpu":    BATCH_SIZE,
        "effective_batch_size":  BATCH_SIZE * NUM_GPUS * NUM_NODES,
    }

    logger = TensorBoardLogger(save_dir=LOG_DIR, name=run_name)
    callbacks = [
        ExperimentMetadataLogger(experiment_payload),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss", mode="min",
            save_top_k=3, save_last=True,
            filename="{epoch:03d}-{val_loss:.4f}",
        ),
        EarlyStopping(
            monitor="val_loss", mode="min",
            patience=EARLY_STOP_PATIENCE_EPOCHS,
            verbose=True,
        ),
        TQDMProgressBar(refresh_rate=20),
    ]

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=NUM_GPUS,
        num_nodes=NUM_NODES,
        strategy="ddp" if (NUM_GPUS * NUM_NODES) > 1 else "auto",
        # Lightning auto-detects SLURM via SLURMEnvironment when SLURM_JOB_ID
        # is set; multi-node DDP "just works" if launched via srun.
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=25,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        # Reproducibility hint — Lightning seeds dataloader RNG per rank
        # so each GPU sees a different shuffle, but the dataset itself is
        # deterministic given SPLIT_SEED.
    )
    trainer.fit(lit, datamodule=dm, ckpt_path=RESUME_CKPT)


if __name__ == "__main__":
    main()
