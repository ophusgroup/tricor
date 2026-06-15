"""DDP training for tricor.macerelax — uses DDPMixin from tricor.ddp.

Refactored from the previous flat script.  All distributed-training plumbing
(process-group init, device pinning, DDP model wrap, size-balanced sampler,
cross-rank collectives, rank-0 I/O) lives in tricor.ddp.DDPMixin; this file
only contains the macerelax-specific trainer: experiment resolution, model
build, training loop, EMA, checkpoint format, TensorBoard logging.

Launch (single or multi-node):

    salloc -A m5241 -C "gpu&hbm80g" -q interactive -t 0:30:00 \\
        --nodes=N --ntasks-per-node=1 --gpus-per-node=4 --gpu-bind=none \\
        --cpus-per-task=64

    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=29500
    export OMP_NUM_THREADS=4

    srun -l torchrun --nnodes=$SLURM_JOB_NUM_NODES \\
         --nproc-per-node=$SLURM_GPUS_PER_NODE \\
         --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \\
         /global/u2/e/ehrdt/tricor/scripts/macerelax/train_perl_ddp.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

# --- data ---
EXPERIMENTS_REGISTRY = "/pscratch/sd/e/ehrdt/tricor/cnos_1e100meV/manifests/experiments.json"
EXPERIMENT_NAME      = "composition_test_big"

CUTOFF                 = 5.0
K_STRIDE_SNAPSHOTS     = 5
ROTATE                 = True
VAL_FRACTION           = 0.05
SPLIT_SEED             = 42
DATA_CACHE_MAX_ENTRIES = 500

# Drop trajectories with n_atoms > this from training/val to avoid OOM-causing
# monster cells.  At ~11,000 (~p95 for composition_test_big) we keep 95% of data.
# Set to None to disable.
MAX_ATOMS_PER_TRAJ     = 11000

# Size-balanced batching: when True, replaces DistributedSampler with one that
# groups similar-sized cells into the same step across ranks.
SIZE_BALANCED_BATCHES  = True

# --- model architecture ---
MAX_Z                       = 120
NODE_DIM                    = 128
EDGE_DIM                    = 128
NUM_CONVS                   = 4
WEIGHT_ENCODER_HIDDEN       = 64
SPECIES_PAIR_DIM            = 16
SHELL_TARGET_SPECIES_DIM    = 8
SHELL_TARGET_HIDDEN         = 64
SHELL_TARGET_DROPOUT        = 0.0
EMA_DECAY                   = 0.9999

# torch.compile: True wraps the model in torch.compile after construction.
# First epoch is slower (compile cost); steady-state ~20–30% faster.
USE_TORCH_COMPILE           = True
TORCH_COMPILE_MODE          = "default"        # "default" | "reduce-overhead" | "max-autotune"
TORCH_COMPILE_DYNAMIC       = True             # tolerate shape variance

# --- optimization ---
LR                          = 1e-3
LR_SCHEDULE                 = "cosine"          # "none" | "cosine"
LR_MIN_RATIO                = 0.01
WARMUP_STEPS                = 500
WEIGHT_DECAY                = 1e-5
GRADIENT_CLIP_VAL           = 1.0

# --- training ---
MAX_EPOCHS                  = 100
BATCH_SIZE                  = 10                # per-rank
NUM_WORKERS                 = 4                 # per-rank dataloader workers
PREFETCH_FACTOR             = 4
PRECISION                   = "bf16"            # "bf16" | "fp32"
CHECK_VAL_EVERY_N_EPOCHS    = 5
EARLY_STOP_PATIENCE_EPOCHS  = 10
LIMIT_TRAIN_BATCHES         = None              # int or None
LIMIT_VAL_BATCHES           = None              # int or None
LOG_EVERY_N_STEPS           = 500

# --- I/O ---
LOG_DIR  = "/pscratch/sd/e/ehrdt/macerelax/lightning_logs"
RUN_NAME = "ddp_v1"                        # tensorboard + checkpoint subdir name
RESUME_CKPT = None                              # path to .pt to resume from, or None

# Allow RESUME_CKPT to be overridden by env var, e.g. set by sbatch's
# auto-resume block (submit_train.sh) or run_interactive.sh.  This is what
# enables chained / interactive runs to pick up from each other's last.pt.
import os as _os
_env_resume = _os.environ.get("RESUME_CKPT_OVERRIDE")
if _env_resume:
    RESUME_CKPT = _env_resume

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import json
import time
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as PyGDataLoader

from tricor.ddp import DDPMixin
from tricor.macerelax.model import RelaxMLModel
from tricor.macerelax.data import RelaxMLDataset, _load_manifest


# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry helpers
# ─────────────────────────────────────────────────────────────────────────────


def resolve_experiment(experiment_name: str, registry_path: str) -> dict:
    path = Path(registry_path)
    with open(path) as f:
        reg = json.load(f)
    exps = reg.get("experiments", {})
    if experiment_name not in exps:
        raise SystemExit(
            f"[abort] experiment {experiment_name!r} not in registry "
            f"{registry_path}.  Available: {sorted(exps)}."
        )
    e = exps[experiment_name]
    base = path.parent
    return {
        "experiment_name":       experiment_name,
        "manifest":              str((base / e["train_manifest"]).resolve()),
        "eval_manifest":         str((base / e["eval_manifest"]).resolve()),
        "description":           e["description"],
        "train_systems":         list(e["train_systems"]),
        "eval_systems":          list(e["eval_systems"]),
        "n_train_trajectories":  int(e["n_train_trajectories"]),
        "n_eval_trajectories":   int(e["n_eval_trajectories"]),
        "registry_generated_at": reg.get("generated_at", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class RelaxMLTrainer(DDPMixin):
    """Hand-written training harness for tricor.macerelax models.

    Encapsulates everything that's specific to this task: experiment
    resolution, model construction (incl. edge_norm freeze, torch.compile),
    EMA buffer maintenance, the train/val loop with bf16 autocast, the
    checkpoint format, and TensorBoard logging.  Inherits DDP plumbing
    from DDPMixin.

    Construct, then call ``.fit()``.
    """

    def __init__(self) -> None:
        # 1. DDP setup — populates self.device, self.world_size, self.global_rank.
        self.setup_distributed()
        self.main_print(
            f"=== DDP init: rank={self.global_rank} "
            f"local_rank={self.local_rank} "
            f"world_size={self.world_size} device={self.device} ==="
        )

        # 2. Resolve experiment + create run directory.
        self.exp = resolve_experiment(EXPERIMENT_NAME, EXPERIMENTS_REGISTRY)
        run_name = RUN_NAME or EXPERIMENT_NAME
        self.run_dir  = Path(LOG_DIR) / run_name / f"run_{int(time.time())}"
        self.ckpt_dir = self.run_dir / "checkpoints"
        if self.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._write_metadata()
        # Wait so all ranks see the freshly-created run_dir before TB/loader reads.
        self.barrier()

        # 3. TensorBoard writer (rank 0 only).
        self.tb: Optional[SummaryWriter] = (
            SummaryWriter(str(self.run_dir / "tb")) if self.is_main else None
        )

        # 4. Precision.
        self.autocast_dtype = torch.bfloat16 if PRECISION == "bf16" else None
        self.main_print(f"[precision] autocast_dtype={self.autocast_dtype}")

    # ── Metadata ────────────────────────────────────────────────────────────

    def _write_metadata(self) -> None:
        payload = {
            "experiment_name":       self.exp["experiment_name"],
            "experiment_desc":       self.exp["description"],
            "manifest":              self.exp["manifest"],
            "eval_manifest":         self.exp["eval_manifest"],
            "train_systems":         self.exp["train_systems"],
            "eval_systems":          self.exp["eval_systems"],
            "n_train_trajectories":  self.exp["n_train_trajectories"],
            "n_eval_trajectories":   self.exp["n_eval_trajectories"],
            "split_seed":            SPLIT_SEED,
            "val_fraction":          VAL_FRACTION,
            "k_stride_snapshots":    K_STRIDE_SNAPSHOTS,
            "cutoff":                CUTOFF,
            "rotate":                ROTATE,
            "max_atoms_per_traj":    MAX_ATOMS_PER_TRAJ,
            "size_balanced_batches": SIZE_BALANCED_BATCHES,
            "shell_target_dropout":  SHELL_TARGET_DROPOUT,
            "ema_decay":             EMA_DECAY,
            "weight_decay":          WEIGHT_DECAY,
            "early_stop_patience":   EARLY_STOP_PATIENCE_EPOCHS,
            "world_size":            self.world_size,
            "batch_size_per_gpu":    BATCH_SIZE,
            "effective_batch_size":  BATCH_SIZE * self.world_size,
            "precision":             PRECISION,
            "torch_compile":         USE_TORCH_COMPILE,
            "torch_compile_mode":    TORCH_COMPILE_MODE if USE_TORCH_COMPILE else None,
        }
        with open(self.run_dir / "experiment.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    # ── Data ────────────────────────────────────────────────────────────────

    def _build_datasets(self) -> tuple[RelaxMLDataset, RelaxMLDataset]:
        rows = _load_manifest(Path(self.exp["manifest"]))
        if not rows:
            raise ValueError(f"Empty manifest at {self.exp['manifest']}")

        if MAX_ATOMS_PER_TRAJ is not None:
            n_before = len(rows)
            rows = [r for r in rows if int(r["n_atoms"]) <= MAX_ATOMS_PER_TRAJ]
            self.main_print(
                f"[filter] kept {len(rows)}/{n_before} trajectories with "
                f"n_atoms ≤ {MAX_ATOMS_PER_TRAJ} (dropped {n_before - len(rows)})"
            )

        n_total = len(rows)
        n_val = max(1, int(round(n_total * VAL_FRACTION)))
        rng = np.random.default_rng(SPLIT_SEED)
        order = rng.permutation(n_total)
        train_rows = [rows[i] for i in order[:-n_val]]
        val_rows   = [rows[i] for i in order[-n_val:]]

        data_root = Path(self.exp["manifest"]).parent
        ds_kwargs = dict(
            cutoff=CUTOFF,
            k_stride_snapshots=K_STRIDE_SNAPSHOTS,
            cache_max_entries=DATA_CACHE_MAX_ENTRIES,
        )
        train_set = RelaxMLDataset(train_rows, data_root, rotate=ROTATE, **ds_kwargs)
        val_set   = RelaxMLDataset(val_rows,   data_root, rotate=False,  **ds_kwargs)

        # Per-sample size key for SizeBalancedDistributedSampler, aligned to
        # __getitem__(idx).  Each sample's atom count = its trajectory's
        # n_atoms (FIRE doesn't add/remove atoms within a trajectory).
        train_set._n_atoms = np.array([
            int(train_rows[traj_idx]["n_atoms"])
            for (traj_idx, _) in train_set._pair_index
        ])
        val_set._n_atoms = np.array([
            int(val_rows[traj_idx]["n_atoms"])
            for (traj_idx, _) in val_set._pair_index
        ])
        return train_set, val_set

    def _build_loaders(self, train_set, val_set):
        train_sizes = train_set._n_atoms if SIZE_BALANCED_BATCHES else None
        val_sizes   = val_set._n_atoms   if SIZE_BALANCED_BATCHES else None

        train_loader, train_sampler = self.make_dataloader(
            train_set,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
            drop_last=True,
            sizes=train_sizes,
            sampler_seed=SPLIT_SEED,
            loader_cls=PyGDataLoader,
            prefetch_factor=PREFETCH_FACTOR,
        )
        val_loader, val_sampler = self.make_dataloader(
            val_set,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            drop_last=False,
            sizes=val_sizes,
            sampler_seed=SPLIT_SEED,
            loader_cls=PyGDataLoader,
            prefetch_factor=PREFETCH_FACTOR,
        )
        return train_loader, val_loader, train_sampler, val_sampler

    # ── Model + EMA ─────────────────────────────────────────────────────────

    def _build_model(self) -> tuple[nn.Module, nn.Module, dict[str, torch.Tensor]]:
        """Build model, take EMA snapshot, optionally compile, wrap in DDP.

        Returns ``(raw_model, ddp_model, ema_state)``.

          * raw_model — the bare nn.Module on this rank's device.  Used for
            EMA reads (state_dict has unprefixed keys) and for checkpoint
            save/load.  Parameters are SHARED with the compiled/DDP-wrapped
            versions.
          * ddp_model — the forward/backward-callable wrapped module.  This
            is what training_step calls into.
          * ema_state — dict of EMA buffer tensors, keyed by unprefixed
            state-dict names, on this rank's device.
        """
        raw_model = RelaxMLModel(
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
        # Freeze the discarded edge_norm — its output isn't connected to the
        # loss (RelaxMLModel.forward drops h_edge), so its grads are always
        # None.  Freezing lets DDP run with find_unused_parameters=False.
        for p in raw_model.processor.edge_norms[-1].parameters():
            p.requires_grad_(False)

        raw_model = raw_model.to(self.device)

        # EMA snapshot taken BEFORE compile so keys are unprefixed.
        ema_state = {
            k: v.detach().clone() for k, v in raw_model.state_dict().items()
        }

        model_for_ddp = raw_model
        if USE_TORCH_COMPILE:
            self.main_print(
                f"[compile] torch.compile(mode={TORCH_COMPILE_MODE!r}, "
                f"dynamic={TORCH_COMPILE_DYNAMIC})"
            )
            model_for_ddp = torch.compile(
                raw_model,
                mode=TORCH_COMPILE_MODE,
                dynamic=TORCH_COMPILE_DYNAMIC,
            )

        ddp_model = self.distribute_model(
            model_for_ddp,
            find_unused_parameters=False,
            broadcast_buffers=False,
            bucket_cap_mb=100,
            gradient_as_bucket_view=True,
        )
        return raw_model, ddp_model, ema_state

    @torch.no_grad()
    def _ema_update(self, ema_state: dict[str, torch.Tensor],
                    raw_model: nn.Module) -> None:
        """In-place EMA update: ema = decay*ema + (1-decay)*raw_model.

        Reads from ``raw_model`` (unwrapped, uncompiled) so the state_dict
        keys match ``ema_state``.  Compile doesn't copy parameters, so
        raw_model's tensors are the same memory the training pass updates.
        """
        decay = EMA_DECAY
        inv = 1.0 - decay
        sd = raw_model.state_dict()
        for k, ema_buf in ema_state.items():
            src = sd[k].detach()
            if not ema_buf.is_floating_point():
                ema_buf.copy_(src)
                continue
            ema_buf.mul_(decay).add_(src.to(ema_buf.dtype), alpha=inv)

    # ── Optimizer + scheduler ───────────────────────────────────────────────

    def _build_optim(self, raw_model: nn.Module, total_steps: int):
        trainable = [p for p in raw_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
        if LR_SCHEDULE == "none":
            return opt, None
        eta_min = LR * LR_MIN_RATIO
        cosine_steps = max(1, total_steps - WARMUP_STEPS)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cosine_steps, eta_min=eta_min,
        )
        if WARMUP_STEPS > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, end_factor=1.0,
                total_iters=WARMUP_STEPS,
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[WARMUP_STEPS],
            )
        else:
            sched = cosine
        return opt, sched

    # ── Forward + loss ──────────────────────────────────────────────────────

    @staticmethod
    def _model_forward(model: nn.Module, batch) -> torch.Tensor:
        return model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
            batch.shell_pair_species, batch.shell_pair_features,
            batch.shell_pair_batch,
            batch.shell_trip_species, batch.shell_trip_features,
            batch.shell_trip_batch,
        )

    def _compute_loss(self, model: nn.Module, batch) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self._model_forward(model, batch)
        loss = (pred - batch.target_displacement).pow(2).sum(dim=-1).mean()
        zero_baseline = batch.target_displacement.pow(2).sum(dim=-1).mean()
        return loss, zero_baseline

    # ── Epoch loops ─────────────────────────────────────────────────────────

    def _train_epoch(
        self, *, epoch, ddp_model, raw_model, ema_state,
        optimizer, scheduler, train_loader, train_sampler,
    ) -> tuple[float, float]:
        ddp_model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        sum_loss = 0.0
        sum_rel  = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if LIMIT_TRAIN_BATCHES is not None and batch_idx >= LIMIT_TRAIN_BATCHES:
                break
            batch = batch.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype,
                                enabled=(self.autocast_dtype is not None)):
                loss, zero_baseline = self._compute_loss(ddp_model, batch)
                rel = loss / zero_baseline.clamp(min=1e-12)

            loss.backward()
            if GRADIENT_CLIP_VAL is not None and GRADIENT_CLIP_VAL > 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), GRADIENT_CLIP_VAL)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            self._ema_update(ema_state, raw_model)

            sum_loss += float(loss.detach().item())
            sum_rel  += float(rel.detach().item())
            n_batches += 1

            if self.is_main and (batch_idx % LOG_EVERY_N_STEPS == 0):
                cur_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                n_total = LIMIT_TRAIN_BATCHES or len(train_loader)
                steps_done = batch_idx + 1
                rate = steps_done / max(elapsed, 1e-9)
                eta_sec = max(0, n_total - steps_done) / max(rate, 1e-9)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                rolling_loss = sum_loss / steps_done
                rolling_rel  = sum_rel  / steps_done
                print(f"  [epoch {epoch} step {batch_idx}/{n_total}] "
                      f"loss={rolling_loss:.4e}  rel={rolling_rel:.3f}  "
                      f"lr={cur_lr:.2e}  rate={rate:.2f} it/s  ETA={eta_str}",
                      flush=True)
                if self.tb is not None:
                    step = epoch * len(train_loader) + batch_idx
                    self.tb.add_scalar("train_loss_step",          loss.item(), step)
                    self.tb.add_scalar("train_relative_loss_step", rel.item(),  step)
                    self.tb.add_scalar("lr",                       cur_lr,      step)

        mean_loss = self.all_reduce_mean(sum_loss / max(1, n_batches))
        mean_rel  = self.all_reduce_mean(sum_rel  / max(1, n_batches))
        if self.is_main:
            dt = time.time() - t0
            print(f"[epoch {epoch}] train_loss={mean_loss:.4e}  "
                  f"train_rel={mean_rel:.3f}  dt={dt:.1f}s", flush=True)
            if self.tb is not None:
                self.tb.add_scalar("train_loss",          mean_loss, epoch)
                self.tb.add_scalar("train_relative_loss", mean_rel,  epoch)
        return mean_loss, mean_rel

    @torch.no_grad()
    def _validate(
        self, *, epoch, ddp_model, val_loader, val_sampler,
    ) -> tuple[float, float]:
        ddp_model.eval()
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        sum_loss = 0.0
        sum_rel  = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(val_loader):
            if LIMIT_VAL_BATCHES is not None and batch_idx >= LIMIT_VAL_BATCHES:
                break
            batch = batch.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype,
                                enabled=(self.autocast_dtype is not None)):
                loss, zero_baseline = self._compute_loss(ddp_model, batch)
                rel = loss / zero_baseline.clamp(min=1e-12)
            sum_loss += float(loss.item())
            sum_rel  += float(rel.item())
            n_batches += 1

        mean_loss = self.all_reduce_mean(sum_loss / max(1, n_batches))
        mean_rel  = self.all_reduce_mean(sum_rel  / max(1, n_batches))
        if self.is_main:
            dt = time.time() - t0
            print(f"[epoch {epoch}]   val_loss={mean_loss:.4e}  "
                  f"val_rel={mean_rel:.3f}  dt={dt:.1f}s", flush=True)
            if self.tb is not None:
                self.tb.add_scalar("val_loss",          mean_loss, epoch)
                self.tb.add_scalar("val_relative_loss", mean_rel,  epoch)
        return mean_loss, mean_rel

    # ── Checkpointing ───────────────────────────────────────────────────────

    def _save_checkpoint(
        self, path, raw_model, ema_state, optimizer, scheduler,
        epoch, best_val, no_improve,
    ) -> None:
        # Note: save the RAW (uncompiled, unwrapped) model's state_dict so
        # the saved keys are unprefixed and portable.
        payload = {
            "model":      raw_model.state_dict(),
            "ema":        ema_state,
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler is not None else None,
            "epoch":      epoch,
            "best_val":   best_val,
            "no_improve": no_improve,
        }
        self.save_on_main(payload, path)

    def _load_checkpoint(self, path, raw_model, ema_state, optimizer, scheduler):
        payload = torch.load(path, map_location=self.device)
        # Strip _orig_mod. prefix if the checkpoint was saved from a compiled model.
        model_state = {
            k.removeprefix("_orig_mod."): v for k, v in payload["model"].items()
        }
        raw_model.load_state_dict(model_state)
        for k, v in payload["ema"].items():
            if k in ema_state:
                ema_state[k].copy_(v.to(ema_state[k].device))
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        return (
            int(payload.get("epoch", 0)),
            float(payload.get("best_val", float("inf"))),
            int(payload.get("no_improve", 0)),
        )

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(self) -> None:
        # Data.
        train_set, val_set = self._build_datasets()
        train_loader, val_loader, train_sampler, val_sampler = \
            self._build_loaders(train_set, val_set)
        self.main_print(
            f"  train_set     : {len(train_set)} samples\n"
            f"  val_set       : {len(val_set)} samples\n"
            f"  train_batches : {len(train_loader)} per rank per epoch\n"
            f"  val_batches   : {len(val_loader)} per rank per epoch"
        )

        # Model + EMA + DDP wrap.
        raw_model, ddp_model, ema_state = self._build_model()

        # Optimizer + scheduler.
        total_steps = MAX_EPOCHS * len(train_loader)
        optimizer, scheduler = self._build_optim(raw_model, total_steps)

        # Optional resume.
        start_epoch = 0
        best_val = float("inf")
        no_improve = 0
        if RESUME_CKPT is not None:
            start_epoch, best_val, no_improve = self._load_checkpoint(
                RESUME_CKPT, raw_model, ema_state, optimizer, scheduler,
            )
            self.main_print(
                f"[resume] loaded {RESUME_CKPT} at epoch {start_epoch}, "
                f"best_val={best_val:.4e}, no_improve={no_improve}"
            )

        # Loop.
        for epoch in range(start_epoch, MAX_EPOCHS):
            self._train_epoch(
                epoch=epoch, ddp_model=ddp_model, raw_model=raw_model,
                ema_state=ema_state, optimizer=optimizer, scheduler=scheduler,
                train_loader=train_loader, train_sampler=train_sampler,
            )

            do_val = (epoch + 1) % CHECK_VAL_EVERY_N_EPOCHS == 0 or epoch == MAX_EPOCHS - 1
            if do_val:
                val_loss, _ = self._validate(
                    epoch=epoch, ddp_model=ddp_model,
                    val_loader=val_loader, val_sampler=val_sampler,
                )
                improved = val_loss < best_val
                if self.is_main:
                    if improved:
                        best_val = val_loss
                        no_improve = 0
                        self._save_checkpoint(
                            self.ckpt_dir / "best.pt",
                            raw_model, ema_state, optimizer, scheduler,
                            epoch, best_val, no_improve,
                        )
                        print(f"[ckpt] new best val_loss={best_val:.4e} → best.pt", flush=True)
                    else:
                        no_improve += 1
                    self._save_checkpoint(
                        self.ckpt_dir / "last.pt",
                        raw_model, ema_state, optimizer, scheduler,
                        epoch, best_val, no_improve,
                    )

                # Sync early-stop decision across ranks.
                stop = self.is_main and (no_improve >= EARLY_STOP_PATIENCE_EPOCHS)
                if self.broadcast_bool(stop):
                    self.main_print(
                        f"[early stop] no improvement in "
                        f"{EARLY_STOP_PATIENCE_EPOCHS} val epochs"
                    )
                    break

        # Cleanup.
        if self.tb is not None:
            self.tb.close()
        self.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    trainer = RelaxMLTrainer()
    trainer.main_print(
        f"[experiment] {trainer.exp['experiment_name']}: "
        f"{trainer.exp['n_train_trajectories']} train trajectories  "
        f"(held out: {trainer.exp['eval_systems']})\n"
        f"  manifest      : {trainer.exp['manifest']}\n"
        f"  BATCH_SIZE={BATCH_SIZE}  WORLD_SIZE={trainer.world_size}  "
        f"effective_batch={BATCH_SIZE * trainer.world_size}\n"
        f"  run_dir       : {trainer.run_dir}"
    )
    trainer.fit()


if __name__ == "__main__":
    main()
