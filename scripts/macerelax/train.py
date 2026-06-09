"""Train the MACE+wall pilot model — fork of scripts/relaxml/train_shelltgt.py.

Same architecture as the relaxml shell_target-conditioned surrogate, but
trained on MACE+wall trajectories from the pilot v1 dataset (see
MACE_RELAX_PILOT.md).  The only meaningful difference is the conditioning
vector: 6 MACE-meaningful fields instead of 9 shell_relax spring-weight
fields.  Architecture, optimization, and training loop are unchanged.

Imports from tricor.macerelax.{model,data} (forked from
tricor.relaxml.{model,data}_shelltgt) so the relaxml production paths
stay frozen.

Before running: ``nvtop`` / ``nvidia-smi`` to see which GPU is free,
set ``GPU_ID`` below, then:

    python scripts/macerelax/train.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these (MUST be before numpy/torch imports)
# ─────────────────────────────────────────────────────────────────────────────

# --- mallard resource caps ---
GPU_ID = 0                # physical GPU index (check with nvidia-smi / nvtop)
NUM_THREADS = 4           # CPU threads; match NUM_WORKERS below.
CHECK_GPU_BUSY = True      # abort if the chosen GPU already has another job

# --- data ---
# Set the experiment name; the train manifest, run name (used for the
# tensorboard log dir + checkpoint dir), and the metadata logged into
# hparams.yaml all derive from it via the registry at
# EXPERIMENTS_REGISTRY.  See scripts/macerelax/generation/make_experiment_manifests.py for the
# canonical EXPERIMENTS declaration.
#
# .npz files referenced by these manifests MUST already contain the four
# shell_target arrays — run scripts/macerelax/generation/add_shell_target_to_pilot.py once after
# trajectory generation completes (idempotent, safe to re-run).
EXPERIMENT_NAME = "polymorph_test"   # one of the keys in EXPERIMENTS
EXPERIMENTS_REGISTRY = "/home/ehrdt/tricor/mace/data/pilot_v1/manifests/experiments.json"

CUTOFF = 5.0              # Å
# Pair stride.  k=1 (single FIRE step per training pair) gave ~0.003 Å
# mean targets — well below the noise floor; train_loss looked pure noise.
# k=10 brings targets to ~0.03 Å mean — comparable to relaxml's per-pair
# scale (which was 5 tricor steps × stiff springs).  Higher k is risky on
# the limited 20 Å receptive field for MACE+wall (cooperative many-body
# forces propagate further per step than shell_relax's local springs).
K_STRIDE_SNAPSHOTS = 5
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
SHELL_TARGET_SPECIES_DIM = 8
SHELL_TARGET_HIDDEN = 64
# Per-graph dropout on the shell_target encoder output during training
# (classifier-free-guidance trick).  0.2 is a sensible default; bump to
# 0.3-0.5 if the model still ignores the conditioning, lower to 0.1 if
# val loss with conditioning suffers.  Set to 0.0 to disable.
# Classifier-free-guidance training dropout on shell_target conditioning.
# v1 was 0.2; v2 set to 0.0 because we don't use the CFG inference trick
# (relaxation surrogates want full conditioning always), and the
# regularization burden moved to weight_decay + early-stopping instead.
SHELL_TARGET_DROPOUT = 0.0
EMA_DECAY = 0.9999
LR = 1e-3
LR_SCHEDULE = "cosine"    # "none" | "cosine"
LR_MIN_RATIO = 0.01
WARMUP_STEPS = 500
# Decoupled L2 via AdamW.  v1 was 0 (vanilla Adam); v2 = 1e-5 to push
# back against the train-vs-val gap that opened up in the v1 run.
WEIGHT_DECAY = 1e-5
# Early-stop patience in EPOCHS.  v1 ran to step 39k and the val
# minimum (0.175) was at step ~27k; it slowly upticked after.  Patience
# of 10 epochs would have caught the val minimum cleanly.
EARLY_STOP_PATIENCE_EPOCHS = 10

# --- training ---
MAX_EPOCHS = 100
BATCH_SIZE = 8            # graphs per batch; each graph ~6k atoms at 50 Å
NUM_WORKERS = 4           # DataLoader processes (match NUM_THREADS above)
LOG_DIR = "./lightning_logs"
# RUN_NAME derives from EXPERIMENT_NAME at runtime so tensorboard /
# checkpoint dirs match the experiment.  Override only for a one-off
# sweep where you want multiple runs of the same experiment.
# v2 variant of composition_test_stride5.  Changes vs v1:
#   - SHELL_TARGET_DROPOUT 0.2 → 0.0  (don't use CFG; want full conditioning)
#   - WEIGHT_DECAY          0   → 1e-5 (target the overfit instead)
#   - EarlyStopping with patience=10 epochs (caught val minimum cleanly)
RUN_NAME = "polymorph_test_stride5_v2_no_st_dropout_wd1e-5"
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

import json
import subprocess
import sys
from pathlib import Path

import torch
torch.set_num_threads(NUM_THREADS)


def resolve_experiment(experiment_name: str, registry_path: str) -> dict:
    """Look up an experiment in the registry written by
    scripts/macerelax/generation/make_experiment_manifests.py.  Returns a dict with:
      - manifest:           absolute path to train manifest
      - eval_manifest:      absolute path to eval manifest
      - description:        free-text doc
      - train_systems:      list of system_id used in training
      - eval_systems:       list of system_id held out for eval
      - n_train_trajectories, n_eval_trajectories
    The resolved dict is suitable for stuffing into Lightning hparams so
    the per-run audit trail (lightning_logs/<name>/version_N/hparams.yaml)
    records exactly what data the model saw.
    """
    path = Path(registry_path)
    with open(path) as f:
        reg = json.load(f)
    exps = reg.get("experiments", {})
    if experiment_name not in exps:
        raise SystemExit(
            f"[abort] experiment {experiment_name!r} not in registry "
            f"{registry_path}.  Available: {sorted(exps)}.  "
            f"Re-run scripts/macerelax/generation/make_experiment_manifests.py after editing its "
            f"EXPERIMENTS config to add new entries."
        )
    e = exps[experiment_name]
    base = path.parent  # manifests/ dir
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
    """At train start, write an ``experiment.json`` file into the
    Lightning version_dir alongside ``hparams.yaml`` and push the same
    payload through the logger's ``log_hyperparams``.  Both records are
    self-contained so the per-run audit trail survives even if the
    EXPERIMENTS config or registry is later edited.
    """

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload

    def on_train_start(self, trainer: L.Trainer,
                        pl_module: L.LightningModule) -> None:
        # 1. File in version_dir for human + grep-friendly inspection.
        log_dir = trainer.log_dir
        if log_dir is not None:
            out = Path(log_dir) / "experiment.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(self.payload, f, indent=2, default=str)
                f.write("\n")
            print(f"[experiment] metadata written to {out}")
        # 2. Logger hparams (lands in tensorboard's "HPARAMS" tab and is
        # also merged into hparams.yaml by the TB logger).  Flatten lists
        # to comma-strings because tensorboard's hparams panel doesn't
        # render sequences.
        if trainer.logger is not None:
            flat = {
                k: (",".join(v) if isinstance(v, (list, tuple)) else v)
                for k, v in self.payload.items()
            }
            trainer.logger.log_hyperparams(flat)

from tricor.macerelax.model import LitRelaxML
from tricor.macerelax.data import RelaxMLDataModule


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

    exp = resolve_experiment(EXPERIMENT_NAME, EXPERIMENTS_REGISTRY)
    run_name = RUN_NAME or EXPERIMENT_NAME
    print(f"[experiment] {exp['experiment_name']}: "
          f"{exp['n_train_trajectories']} train trajectories"
          f"  (held out: {exp['eval_systems']})")
    print(f"  train manifest: {exp['manifest']}")
    print(f"  eval  manifest: {exp['eval_manifest']}")

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

    # torch.compile the inner model for ~1.5–2x throughput on Ampere/Hopper.
    # dynamic=True so we don't pay recompilation cost when edge counts vary
    # slightly between batches (PBC neighbor lists drift with atom positions).
    # The first epoch will be slower than steady-state due to compilation.
    # The ShellTargetEncoder's forward is decorated with @_dynamo.disable in
    # model_shelltgt.py so its variable-length per-pair / per-triplet
    # tensors don't trigger repeated recompiles of the surrounding model.
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
        # Inference-relevant training settings — eval reads these back to
        # derive MAX_ITER, build the right edge graph, etc.
        "k_stride_snapshots":    K_STRIDE_SNAPSHOTS,
        "cutoff":                CUTOFF,
        "rotate":                ROTATE,
        # Regularization knobs — record so future runs can be compared.
        "shell_target_dropout":  SHELL_TARGET_DROPOUT,
        "weight_decay":          WEIGHT_DECAY,
        "early_stop_patience":   EARLY_STOP_PATIENCE_EPOCHS,
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
        # Stop training once val_loss hasn't improved for
        # EARLY_STOP_PATIENCE_EPOCHS consecutive validation rounds.  Catches
        # the val minimum and avoids the post-minimum overfit drift we saw
        # in v1 (val_loss bottomed at ~0.175 and crept back up to ~0.22).
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
